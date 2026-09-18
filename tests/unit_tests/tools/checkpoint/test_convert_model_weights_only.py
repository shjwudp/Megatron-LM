# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
GPU unit test for ``convert_checkpoint(..., model_weights_only=True)``.

``tools/checkpoint/checkpoint_inspector.py``'s
``convert-torch-dist-to-fsdp-dtensor`` command converts a ``torch_dist``
checkpoint to ``fsdp_dtensor`` for Megatron-FSDP v2. M-FSDP v2 does not
implement optimizer checkpointing, so a converted checkpoint that still
carries optimizer state cannot be loaded. The ``--model-weights-only`` flag
(``model_weights_only=True``) drops all optimizer state and emits only model
weights plus the non-optimizer common state (``args``, ``iteration``,
``checkpoint_version``).

The test guards two confirmed defects in that converter:

1. **Common-state format.** The converter used to call the deprecated
   ``strategies.common.load_common`` directly, which only understands the
   legacy ``common.pt`` layout. Current Megatron never calls ``save_common``,
   so it stores the common state as the DCP object
   ``common_state/shard_0_1`` and the converter crashed with
   ``CheckpointingException: Common file .../common.pt does not exist`` on
   every current checkpoint. Both layouts are built here and both are passed
   through the format-agnostic ``load_common_state_dict``.

2. **Distributed-optimizer keys.** With ``use_distributed_optimizer: true``
   the real optimizer keys are nested under the chained optimizer
   (``chained_<i>.optimizer...``). A bare ``key.startswith("optimizer.")``
   filter therefore missed them, so they were loaded, rewritten as
   ``model.module.chained_<i>.optimizer.*`` and emitted -- a "weights-only"
   checkpoint that still carried optimizer state, while a naive "no key starts
   with ``optimizer.``" assertion still passed.

This test builds a synthetic ``torch_dist`` checkpoint that genuinely contains
optimizer state in both the non-chained (``optimizer.state.<slot>.<param>``)
and the chained distributed-optimizer (``chained_<i>.optimizer.<...>``)
layouts, in each of the two common-state formats, runs the converter in both
modes, and asserts:

  * ``model_weights_only=True`` -> every model weight is present and
    element-wise identical, and **no** output key matches the dotted-component
    predicate ``(^|\\.)optimizer(\\.|$)``.
  * ``model_weights_only=False`` (default) -> the same model weights
    round-trip, and the output still carries optimizer keys, proving the flag
    is what makes the difference.
  * the non-optimizer common state survives in both modes.
  * the source fixture really contained optimizer keys (tensor *and*
    common-state), so the "no optimizer keys" assertion cannot pass vacuously.
  * ``load_common_state_dict`` reads both the legacy ``common.pt`` and the
    current ``ShardedObject("common_state")`` layout.

The conversion cases are parametrized over the two common-state layouts, so
each mode is exercised against both real-world checkpoint formats.

The conversion cases are GPU tests: ``convert_checkpoint`` builds a CUDA
``DeviceMesh`` and allocates CUDA DTensors, so an NCCL process group and at
least one GPU are required (the CLI's ``init_process_group`` also forces the
NCCL backend). They therefore cannot run on a CPU-only machine. The
``test_is_optimizer_key_matches_dotted_components`` predicate check is pure and
needs no GPU, so the DEFECT 2 filter regression is covered even where the
end-to-end conversion cannot run.

Launch it (all ranks participate in the DCP collectives; the default
``torch.distributed.run`` unit-test convention applies, and it is intended for
an H100-class GPU platform):

    uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \\
        tests/unit_tests/tools/checkpoint/test_convert_model_weights_only.py

or standalone on a shared filesystem (``--output-root`` must be visible from
every rank; single-node runs can use the default ``/tmp``-based root):

    torchrun --nproc_per_node=8 \\
        tests/unit_tests/tools/checkpoint/test_convert_model_weights_only.py \\
        --output-root /shared/scratch/ckpt_model_weights_only
"""

import argparse
import io
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

# Make the conversion tool and helpers importable, matching the sibling tests.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)

# Override to point the shared output root at a shared filesystem (multi-node).
_SHARED_ROOT_ENV = 'MCORE_CKPT_MODEL_WEIGHTS_ONLY_ROOT'

# The predicate the converter must use for weights-only filtering: `optimizer`
# as a *dotted path component*. It is deliberately not the same as a bare
# `startswith('optimizer.')` test: distributed-optimizer keys are prefixed with
# `chained_<i>.` and would slip past the prefix test.
_OPTIMIZER_KEY_RE = re.compile(r'(^|\.)optimizer(\.|$)')

# The two real-world layouts of a torch_dist checkpoint's common state.
_COMMON_STATE_FORMATS = ('current', 'legacy')


def _optimizer_keys(keys):
    """Keys containing ``optimizer`` as a dotted path component."""
    return sorted(k for k in keys if _OPTIMIZER_KEY_RE.search(k))


def _log(rank, message):
    print(f"[rank={rank}] {message}", flush=True)


def _shared_root():
    """Directory shared by every rank (DCP saves/loads are collective)."""
    return os.environ.get(
        _SHARED_ROOT_ENV,
        os.path.join(tempfile.gettempdir(), 'mcore_ckpt_model_weights_only_test'),
    )


def _ensure_nccl_process_group():
    """Initialize the default NCCL process group if the harness has not.

    Returns ``True`` when a usable CUDA/NCCL group is available. A pre-existing
    non-NCCL default group (e.g. a gloo group initialized by an earlier test in
    the same pytest session) cannot back a CUDA ``DeviceMesh``, so it is
    reported as unusable rather than failing obscurely.
    """
    if dist.is_initialized():
        return dist.get_backend() == 'nccl'
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    dist.init_process_group(backend='nccl')
    return True


# ---------------------------------------------------------------------------
# Synthetic checkpoint fixtures
# ---------------------------------------------------------------------------


def _build_model_state_dict(num_layers, hidden_size, vocab_size, dtype):
    """Deterministic bare-key GPT state dict (identical on every rank)."""
    torch.manual_seed(0x5EED)
    sd = OrderedDict()
    sd['embedding.word_embeddings.weight'] = torch.randn(
        vocab_size, hidden_size, dtype=dtype
    )
    for i in range(num_layers):
        p = f'decoder.layers.{i}.'
        sd[p + 'input_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'self_attention.linear_qkv.weight'] = torch.randn(
            3 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'self_attention.linear_proj.weight'] = torch.randn(
            hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'pre_mlp_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
            4 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
            hidden_size, 4 * hidden_size, dtype=dtype
        )
    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)
    return sd


def _build_optimizer_state_dict(model_state_dict, num_chained=2):
    """Adam state covering both the non-chained and the distributed layouts.

    The non-chained family, ``optimizer.state.<slot>.<param>``, is the layout
    the converter's optimizer-conversion path understands (and what the old
    fixture exercised):

        ``tests/unit_tests/dist_checkpointing/test_optimizer.py`` uses
        ``optimizer.state.{exp_avg,exp_avg_sq}.<layer_name>``.

    The chained family is what ``use_distributed_optimizer: true`` actually
    produces: a ``ChainedOptimizer`` prefixes every member optimizer's sharded
    state with ``chained_<i>.`` (``ChainedOptimizer.sharded_state_dict`` in
    ``megatron/core/optimizer/optimizer.py``), and ``DistributedOptimizer``
    keys its state as ``optimizer.state.<slot>.<param>`` or wraps it under
    ``optimizer.distributed.dp_group_idx_<dp>``
    (``megatron/core/optimizer/distrib_optimizer.py``). No such key starts with
    the literal ``optimizer.`` prefix, which is exactly the shape the old
    prefix filter missed.
    """
    torch.manual_seed(0x0D15)
    optimizer_sd = OrderedDict()
    for model_key, tensor in model_state_dict.items():
        for slot in ('exp_avg', 'exp_avg_sq'):
            optimizer_sd[f'optimizer.state.{slot}.{model_key}'] = torch.randn_like(
                tensor
            )
    for i in range(num_chained):
        state_prefix = f'chained_{i}.optimizer.state'
        bucket_prefix = (
            f'chained_{i}.optimizer.distributed.dp_group_idx_{i}.param_state'
        )
        for model_key, tensor in model_state_dict.items():
            for slot in ('exp_avg', 'exp_avg_sq'):
                optimizer_sd[f'{state_prefix}.{slot}.{model_key}'] = torch.randn_like(
                    tensor
                )
                optimizer_sd[f'{bucket_prefix}.{slot}.{model_key}'] = torch.randn_like(
                    tensor
                )
    return optimizer_sd


def _build_ckpt_args(num_layers, hidden_size, vocab_size):
    """Checkpoint args as a plain ``dict``.

    Real checkpoints pass through
    ``megatron.training.training.preprocess_common_state_dict``, which stores
    ``vars(args)``. A plain dict is also required by the *current* common-state
    layout, which ``load_common_state_dict`` reads back with
    ``torch.load(..., weights_only=True)``; a ``SimpleNamespace`` would not
    unpickle in restricted mode. The converter flattens this dict, so the
    surviving keys are ``args.<field>``.
    """
    return dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=hidden_size * 4,
        seq_length=256,
        max_position_embeddings=256,
        iteration=100,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        train_iters=1000,
        train_samples=0,
        tokenizer_type='GPT2BPETokenizer',
        position_embedding_type='rope',
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        num_moe_experts=None,
        moe_shared_expert_intermediate_size=None,
        moe_layer_freq=1,
        vocab_size=vocab_size,
    )


def _build_common_state(num_layers, hidden_size, vocab_size, iteration=100, num_chained=2):
    """Common (non-sharded) state: non-tensor state plus optimizer key families.

    ``optimizer.param_groups`` is the non-chained layout the converter reads via
    ``common_state["optimizer"]["param_groups"]``; ``flatten`` turns it into
    ``optimizer.param_groups.<i>.<field>`` entries that the default conversion
    merges into the output (and that the old fixture exercised).

    The ``chained_<i>.optimizer.*`` entries mimic the distributed-optimizer
    *common* state, which the converter flattens and merges at the very end of
    ``convert_checkpoint`` (the second weights-only filter site). They are
    non-tensor values, so they never appear in the DCP metadata: only the
    ``common.pt`` / ``ShardedObject("common_state")`` payload carries them.
    """
    common_state = {
        'args': _build_ckpt_args(num_layers, hidden_size, vocab_size),
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'optimizer': {
            'param_groups': [
                {'lr': 1.0e-4, 'weight_decay': 0.01, 'betas': [0.9, 0.95]}
            ]
        },
    }
    for i in range(num_chained):
        opt_prefix = f'chained_{i}.optimizer.distributed.dp_group_idx_{i}'
        common_state[f'{opt_prefix}.optimizer'] = {
            'param_groups': [
                {'lr': 1.0e-4, 'weight_decay': 0.01, 'betas': [0.9, 0.95]}
            ]
        }
        common_state[f'{opt_prefix}.param_state_sharding_type'] = (
            'fully_sharded_bucket_space'
        )
    return common_state


# ---------------------------------------------------------------------------
# Source-checkpoint writers (one per common-state layout)
# ---------------------------------------------------------------------------


def _save_source_checkpoint(
    full_sd, common_state, save_dir, common_state_format='current', model_prefix=''
):
    """Write a synthetic ``torch_dist`` source checkpoint in a chosen layout.

    ``'legacy'``
        Uses ``dist_checkpoint_io.save_dist_checkpoint_full``, which calls the
        deprecated ``strategies.common.save_common`` and therefore writes a
        separate ``common.pt`` file. This is the layout the old test fixture
        exercised and the only layout ``load_common`` understands.

    ``'current'``
        What current Megatron writes: ``save_common`` is called nowhere under
        ``megatron/``, so the common state is stored as the DCP object
        ``common_state/shard_0_1`` (``ShardedObject("common_state", ...)
        .unique_key``) whose payload is ``torch.save([common_state_dict])`` --
        exactly what ``dist_checkpointing.serialization.save`` writes. There is
        no ``common.pt``, so ``load_common`` alone raises
        ``CheckpointingException`` on this layout.
    """
    from checkpoint_inspector import save_checkpoint_with_pickle_protocol

    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    os.makedirs(save_dir, exist_ok=True)

    if common_state_format == 'legacy':
        from dist_checkpoint_io import save_dist_checkpoint_full

        save_dist_checkpoint_full(
            full_sd,
            common_state,
            save_dir,
            model_prefix=model_prefix,
            backend='torch_dist',
        )
        return

    assert common_state_format == 'current', common_state_format

    raw_state_dict = OrderedDict()
    for bare_key, tensor in full_sd.items():
        full_key = f'{model_prefix}{bare_key}' if model_prefix else bare_key
        raw_state_dict[full_key] = tensor.contiguous()

    # The current layout stores the whole common state dict as a single
    # ShardedObject. `_mcore_to_torch_sharded_object` serializes it as a
    # one-element list, which is why `load_common_state_dict` returns
    # `loaded[0]`.
    payload = io.BytesIO()
    torch.save([common_state], payload)
    payload.seek(0)
    common_key = ShardedObject('common_state', None, (1,), (0,)).unique_key
    assert common_key == 'common_state/shard_0_1', common_key
    raw_state_dict[common_key] = payload

    save_checkpoint_with_pickle_protocol(raw_state_dict, save_dir)
    if dist.get_rank() == 0:
        save_config(CheckpointingConfig(sharded_backend='torch_dist'), save_dir)
    dist.barrier()


# ---------------------------------------------------------------------------
# Conversion + verification
# ---------------------------------------------------------------------------


def _metadata_keys(ckpt_dir):
    """Keys recorded in a (raw) DCP checkpoint's metadata."""
    return set(FileSystemReader(ckpt_dir).read_metadata().state_dict_metadata.keys())


def _load_full_tensors(ckpt_dir):
    """Load a raw DCP checkpoint into full, gathered CPU tensors.

    ``convert_checkpoint`` writes a *raw* DCP checkpoint through
    ``torch.distributed.checkpoint`` and does not write Megatron's
    ``metadata.json``, so ``dist_checkpoint_io.load_dist_checkpoint_full``
    (which requires that config) cannot read it. This mirrors that helper's
    tensor-loading core, minus the config/prefix/filter logic.
    """
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()
    state_dict = {}
    for key, md in metadata.state_dict_metadata.items():
        if not isinstance(md, TensorStorageMetadata):
            continue
        state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype, device='cpu')
    dcp.load(state_dict, storage_reader=reader, planner=DefaultLoadPlanner())
    return state_dict


def run_case(
    label,
    model_weights_only,
    output_root,
    num_layers=2,
    hidden_size=32,
    vocab_size=64,
    dtype=torch.float32,
    common_state_format='current',
):
    """Build a torch_dist ckpt with optimizer state, convert, and verify."""
    from checkpoint_inspector import convert_checkpoint, flatten

    from megatron.core.dist_checkpointing.serialization import load_common_state_dict

    rank = dist.get_rank()
    case_dir = os.path.join(output_root, f'{label}_{common_state_format}')
    src_dir = os.path.join(case_dir, 'torch_dist_src', 'iter_0000100')
    dst_dir = os.path.join(case_dir, 'fsdp_dtensor_dst')

    if rank == 0 and os.path.isdir(case_dir):
        shutil.rmtree(case_dir, ignore_errors=True)
    dist.barrier()
    os.makedirs(os.path.dirname(src_dir), exist_ok=True)
    dist.barrier()

    model_sd = _build_model_state_dict(num_layers, hidden_size, vocab_size, dtype)
    full_sd = OrderedDict(model_sd)
    full_sd.update(_build_optimizer_state_dict(model_sd))
    common_state = _build_common_state(num_layers, hidden_size, vocab_size)

    # ``model_prefix=''`` keeps the bare model keys the converter expects and
    # leaves the chained optimizer keys exactly as Megatron shards them.
    _save_source_checkpoint(
        full_sd,
        common_state,
        src_dir,
        common_state_format=common_state_format,
        model_prefix='',
    )
    dist.barrier()

    # Prove the fixture really uses the requested layout: only the legacy
    # layout has a common.pt, and only the current layout has the internal
    # ShardedObject metadata key.
    has_common_pt = os.path.exists(os.path.join(src_dir, 'common.pt'))
    assert has_common_pt == (common_state_format == 'legacy'), (
        f"[{label}] common.pt presence does not match format "
        f"'{common_state_format}'"
    )
    src_keys = _metadata_keys(src_dir)
    assert ('common_state/shard_0_1' in src_keys) == (
        common_state_format == 'current'
    ), f"[{label}] unexpected common-state metadata key layout: {sorted(src_keys)[:5]}"

    # DEFECT 1 guard: the format-agnostic accessor must read *both* layouts.
    # The converter used to call load_common() directly, which only handles
    # legacy checkpoints and raised CheckpointingException on current ones.
    loaded_common = load_common_state_dict(src_dir)
    assert loaded_common['iteration'] == 100, (
        f"[{label}] load_common_state_dict lost 'iteration' for "
        f"'{common_state_format}' layout"
    )
    assert 'args' in loaded_common, (
        f"[{label}] load_common_state_dict lost 'args' for "
        f"'{common_state_format}' layout"
    )

    # DEFECT 2 guard, part 1: the source must actually contain optimizer keys
    # in both the tensor metadata and the (flattened) common state, otherwise
    # the "no optimizer keys in the output" assertion below passes vacuously.
    src_optimizer_keys = _optimizer_keys(src_keys)
    assert src_optimizer_keys, (
        f"[{label}] fixture metadata has no key matching "
        f"{_OPTIMIZER_KEY_RE.pattern!r}: {sorted(src_keys)[:5]}"
    )
    src_common_optimizer_keys = _optimizer_keys(flatten(loaded_common).keys())
    assert src_common_optimizer_keys, (
        f"[{label}] fixture common state has no key matching "
        f"{_OPTIMIZER_KEY_RE.pattern!r}"
    )
    _log(
        rank,
        f"[{label}/{common_state_format}] source has "
        f"{len(src_optimizer_keys)} optimizer tensor keys and "
        f"{len(src_common_optimizer_keys)} optimizer common-state keys",
    )

    # Every rank must participate: dcp.load / _save_state_dict are collective.
    convert_checkpoint(
        src_dir,
        dst_dir,
        False,
        process_group=dist.group.WORLD,
        model_weights_only=model_weights_only,
    )
    dist.barrier()

    dst_keys = _metadata_keys(dst_dir)
    optimizer_keys = _optimizer_keys(dst_keys)
    for expected_common_key in ('iteration', 'checkpoint_version'):
        assert expected_common_key in dst_keys, (
            f"[{label}] non-optimizer common state '{expected_common_key}' was dropped"
        )
    # ``args`` is flattened by the converter, so it survives as ``args.<field>``.
    assert any(k == 'args' or k.startswith('args.') for k in dst_keys), (
        f"[{label}] non-optimizer common state 'args' was dropped"
    )

    # Model weights must survive element-wise, without depending on load order.
    loaded = _load_full_tensors(dst_dir)
    model_prefix = 'model.module.'
    recovered = {
        k[len(model_prefix):]: v
        for k, v in loaded.items()
        if k.startswith(model_prefix)
    }
    missing = [k for k in model_sd if k not in recovered]
    mismatch = [
        k
        for k in model_sd
        if k in recovered and not torch.equal(model_sd[k], recovered[k].to(model_sd[k].dtype))
    ]
    assert not missing, f"[{label}] missing model weights: {missing[:5]}"
    assert not mismatch, f"[{label}] mismatched model weights: {mismatch[:5]}"

    if model_weights_only:
        # This is the assertion that catches DEFECT 2: it is evaluated over the
        # whole output key set with the dotted-component predicate, so
        # `model.module.chained_<i>.optimizer.*` (and any merged
        # `chained_<i>.optimizer.distributed.*` common key) fails it, whereas a
        # bare `startswith('optimizer.')` check would have passed.
        assert not optimizer_keys, (
            f"[{label}] model-weights-only output still has optimizer keys "
            f"(predicate {_OPTIMIZER_KEY_RE.pattern!r}): {optimizer_keys[:5]}"
        )
        _log(
            rank,
            f"[{label}/{common_state_format}] PASS: {len(recovered)} model weights "
            f"round-tripped, 0 optimizer keys",
        )
    else:
        assert optimizer_keys, (
            f"[{label}] default conversion unexpectedly dropped all optimizer keys"
        )
        _log(
            rank,
            f"[{label}/{common_state_format}] PASS: {len(recovered)} model weights "
            f"round-tripped, {len(optimizer_keys)} optimizer keys preserved",
        )

    dist.barrier()


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _require_cuda_nccl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required: convert_checkpoint builds a CUDA DeviceMesh.")
    if not _ensure_nccl_process_group():
        pytest.skip(
            "Requires an NCCL process group; launch via "
            "'torch.distributed.run --nproc-per-node <N> -m pytest' or torchrun."
        )
    yield
    if dist.is_initialized():
        dist.barrier()


@pytest.mark.parametrize(
    'key, expected',
    [
        # `optimizer` as a dotted path component -> filtered in weights-only mode.
        ('optimizer', True),
        ('optimizer.state.exp_avg.weight', True),
        ('chained_0.optimizer.state.exp_avg.embedding.weight', True),
        ('chained_1.optimizer.distributed.dp_group_idx_1.param_state.exp_avg.w', True),
        ('chained_1.optimizer.optimizer.param_groups.0.lr', True),
        ('model.module.optimizer', True),
        # `optimizer` only as a substring / longer word -> a model weight.
        ('myoptimizer.weight', False),
        ('optimizerizer.weight', False),
        ('args.use_distributed_optimizer', False),
        ('model.module.mlp.optimizers.0.weight', False),
        ('iteration', False),
        ('checkpoint_version', False),
    ],
)
def test_is_optimizer_key_matches_dotted_components(key, expected):
    """The weights-only filter must match `optimizer` as a *dotted component*.

    This is the pure-helper regression test for DEFECT 2 and needs neither CUDA
    nor a process group. Distributed-optimizer state is nested under the chained
    optimizer (``chained_<i>.optimizer...``), so the old
    ``key.startswith('optimizer.')`` predicate missed it and the "weights-only"
    checkpoint kept optimizer state while a naive prefix assertion still passed.
    """
    from checkpoint_inspector import is_optimizer_key

    assert is_optimizer_key(key) is expected


@pytest.mark.usefixtures('_require_cuda_nccl')
@pytest.mark.parametrize('common_state_format', _COMMON_STATE_FORMATS)
def test_model_weights_only_drops_all_optimizer_state(common_state_format):
    """--model-weights-only emits model.* only; no optimizer key of any kind."""
    run_case(
        'model_weights_only',
        True,
        _shared_root(),
        common_state_format=common_state_format,
    )


@pytest.mark.usefixtures('_require_cuda_nccl')
@pytest.mark.parametrize('common_state_format', _COMMON_STATE_FORMATS)
def test_default_conversion_keeps_optimizer_state(common_state_format):
    """The default path is unchanged and still carries optimizer entries."""
    run_case(
        'default',
        False,
        _shared_root(),
        common_state_format=common_state_format,
    )


# ---------------------------------------------------------------------------
# Standalone entry point (torchrun)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='GPU test for checkpoint_inspector --model-weights-only.'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Shared-filesystem directory visible from every rank. Defaults '
        'to a /tmp-based directory (single node only).',
    )
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit('ERROR: CUDA is required (convert_checkpoint builds a CUDA DeviceMesh).')
    if not _ensure_nccl_process_group():
        sys.exit(
            'ERROR: could not initialize an NCCL process group; launch with '
            'torchrun/torch.distributed.run on a GPU node.'
        )

    output_root = os.path.abspath(args.output_root or _shared_root())
    size_kwargs = dict(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )
    for common_state_format in _COMMON_STATE_FORMATS:
        run_case(
            'model_weights_only',
            True,
            output_root,
            common_state_format=common_state_format,
            **size_kwargs,
        )
        run_case(
            'default',
            False,
            output_root,
            common_state_format=common_state_format,
            **size_kwargs,
        )

    if dist.get_rank() == 0:
        _log(
            0,
            'PASS: --model-weights-only drops all optimizer state; default keeps '
            'it (both legacy and current common-state layouts).',
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
