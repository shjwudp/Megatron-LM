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

The test guards three confirmed defects in that converter:

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

3. **The internal common-state blob.** The metadata loop used to copy the
   source's DCP-internal ``common_state/shard_0_1`` object into the output
   verbatim. Its key name contains no ``optimizer`` component, so every
   key-level assertion passed, but its *payload* is the whole, unfiltered
   common state dict -- optimizer entries included -- i.e. exactly the
   DEFECT-2 failure mode one level down. The converter now drops that internal
   key under ``--model-weights-only``.

This test builds a synthetic ``torch_dist`` checkpoint that genuinely contains
optimizer state in both the non-chained (``optimizer.state.<slot>.<param>``)
and the chained distributed-optimizer (``chained_<i>.optimizer.<...>``)
layouts, in each of the two common-state formats, runs the converter in both
modes, and asserts:

  * ``model_weights_only=True`` -> every model weight is present and
    element-wise identical; **no** output key matches the dotted-component
    predicate ``(^|\\.)optimizer(\\.|$)``; no *bytes payload* of any output
    entry hides such a key either (the blob scan); and the internal
    ``common_state/shard_0_1`` blob is not re-emitted at all.
  * ``model_weights_only=False`` (default) -> the same model weights
    round-trip, and the output still carries optimizer keys, proving the flag
    is what makes the difference.
  * the non-optimizer common state survives in both modes as individual
    ``args`` / ``iteration`` / ``checkpoint_version`` keys -- the exact three
    keys rank 0 of the real ``fsdp_dtensor`` loader probes for with a raw DCP
    load (``megatron/training/checkpointing.py``).
  * the source fixture really contained optimizer keys (tensor, common-state
    *and*, for the current layout, inside the blob payload), so the negative
    assertions cannot pass vacuously.
  * ``load_common_state_dict`` reads both the legacy ``common.pt`` and the
    current ``ShardedObject("common_state")`` layout.

Note on the converted output: it is a *raw* DCP checkpoint and the converter
does not write Megatron's ``metadata.json``, so ``load_common_state_dict``
cannot read it (it fails in ``verify_checkpoint`` before ever looking for a
common-state object). That is pre-existing and independent of this fix; the
``fsdp_dtensor`` loader does not call that accessor -- ``_probe_loader_common_keys``
mirrors what it actually does.

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
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

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

# DCP-internal key of the current layout's common-state ShardedObject. It is
# `ShardedObject("common_state", None, (1,), (0,)).unique_key` -- the same key
# `load_common_state_dict` derives -- and its payload is the whole, unfiltered
# common state dict (optimizer entries included).
_COMMON_STATE_DCP_KEY = 'common_state/shard_0_1'


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
    """Checkpoint args as an ``argparse.Namespace``, exactly as saved.

    ``generate_state_dict`` stores ``state_dict['args'] = args`` (a Namespace),
    and the loader reads it back as an object
    (``megatron/training/checkpointing.py``: ``checkpoint_args = state_dict['args']``
    followed by ``hasattr``/``getattr``). The converter's ``flatten`` treats a
    Namespace as a leaf, so the converted checkpoint keeps a single top-level
    ``args`` key -- which is what the ``fsdp_dtensor`` loader probes for.
    ``Namespace`` (and ``SimpleNamespace``) is allowlisted for
    ``torch.load(weights_only=True)`` by ``megatron/core/safe_globals.py``.
    """
    return SimpleNamespace(
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


def _load_bytes_payloads(ckpt_dir):
    """Load every ``BytesStorageMetadata`` entry of a raw DCP checkpoint.

    Returns ``{dcp_key: payload}``; tensor entries are ignored. DCP's default
    load planner normally deserializes a bytes entry for us (its
    ``load_bytes`` calls ``torch.load``), but older releases hand back the raw
    ``io.BytesIO`` instead -- ``_keys_in_object`` copes with either.
    """
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()
    targets = {
        key: io.BytesIO()
        for key, md in metadata.state_dict_metadata.items()
        if isinstance(md, BytesStorageMetadata)
    }
    if targets:
        dcp.load(targets, storage_reader=reader, planner=DefaultLoadPlanner())
    return targets


def _keys_in_object(obj, _depth=0):
    """Every string key reachable inside a deserialized checkpoint payload.

    The internal ``common_state`` blob is a pickled ``io.BytesIO`` whose buffer
    is itself a pickled ``[common_state_dict]`` list, so ``BytesIO`` payloads
    are followed recursively. ``weights_only=False`` is intentional: these
    payloads are produced by this test itself, and the check must be able to
    see through the double serialization the converter used to apply.
    """
    if _depth > 4:
        return []
    if isinstance(obj, io.BytesIO):
        try:
            inner = torch.load(io.BytesIO(obj.getvalue()), weights_only=False)
        except Exception:
            return []
        return _keys_in_object(inner, _depth + 1)
    if isinstance(obj, dict):
        keys = []
        for k, v in obj.items():
            if isinstance(k, str):
                keys.append(k)
            keys.extend(_keys_in_object(v, _depth + 1))
        return keys
    if isinstance(obj, (list, tuple)):
        keys = []
        for v in obj:
            keys.extend(_keys_in_object(v, _depth + 1))
        return keys
    return []


def _optimizer_keys_in_bytes_payloads(ckpt_dir):
    """Optimizer-component keys hidden *inside* the payload of bytes entries.

    Tensor keys are covered by ``_metadata_keys``. This catches the same class
    of leak one level down, where the offending name is not the DCP entry name
    but a string inside a serialized blob: the internal ``common_state`` object
    is exactly that, since its payload is the whole, unfiltered common state
    dict, optimizer entries included.
    """
    hidden = []
    for _key, payload in _load_bytes_payloads(ckpt_dir).items():
        hidden.extend(_optimizer_keys(_keys_in_object(payload)))
    return sorted(hidden)


def _probe_loader_common_keys(ckpt_dir):
    """Read the converted checkpoint the way the real fsdp_dtensor loader does.

    ``megatron/training/checkpointing.py`` loads the non-tensor state of an
    ``fsdp_dtensor`` checkpoint on rank 0 with a raw DCP load of exactly
    ``{'args', 'iteration', 'checkpoint_version'}``; ``args`` must come back as
    an object (the loader does ``state_dict['args']`` and then ``hasattr``).
    """
    state_dict = {'args': None, 'iteration': None, 'checkpoint_version': None}
    dcp.load(state_dict=state_dict, checkpoint_id=ckpt_dir)
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
    assert (_COMMON_STATE_DCP_KEY in src_keys) == (
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
    # in the tensor metadata, the (flattened) common state, and -- for the
    # current layout -- the payload of the internal common-state blob. Without
    # these, the "no optimizer keys in the output" assertions pass vacuously.
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
    src_blob_optimizer_keys = _optimizer_keys_in_bytes_payloads(src_dir)
    if common_state_format == 'current':
        # The current layout's `common_state/shard_0_1` payload is the whole,
        # unfiltered common state dict, so it must expose the chained optimizer
        # entries. Post-fix the converter must not re-emit that blob at all.
        assert src_blob_optimizer_keys, (
            f"[{label}] fixture common_state blob payload exposes no key matching "
            f"{_OPTIMIZER_KEY_RE.pattern!r}; the blob scan would be vacuous"
        )
    _log(
        rank,
        f"[{label}/{common_state_format}] source has "
        f"{len(src_optimizer_keys)} optimizer tensor keys, "
        f"{len(src_common_optimizer_keys)} optimizer common-state keys and "
        f"{len(src_blob_optimizer_keys)} optimizer keys inside bytes payloads",
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
    hidden_optimizer_keys = _optimizer_keys_in_bytes_payloads(dst_dir)
    for expected_common_key in ('args', 'iteration', 'checkpoint_version'):
        assert expected_common_key in dst_keys, (
            f"[{label}] non-optimizer common state '{expected_common_key}' was dropped"
        )

    # Loadability: read the converted checkpoint the way the real fsdp_dtensor
    # loader does (raw DCP load of the three non-tensor keys, `args` as an
    # object). See `_probe_loader_common_keys`.
    #
    # `load_common_state_dict(dst_dir)` is deliberately *not* used here: the
    # converter writes a raw DCP checkpoint with no Megatron `metadata.json`,
    # so that accessor fails in `verify_checkpoint` before it ever looks at a
    # common-state object. The fsdp_dtensor loader does not call it either.
    loader_state = _probe_loader_common_keys(dst_dir)
    assert loader_state['args'] is not None, f"[{label}] converted ckpt has no 'args'"
    assert getattr(loader_state['args'], 'hidden_size', None) == hidden_size, (
        f"[{label}] converted ckpt 'args' is not the checkpoint args object"
    )
    assert loader_state['iteration'] == 100, f"[{label}] converted ckpt 'iteration' lost"
    assert loader_state['checkpoint_version'] == 3.0, (
        f"[{label}] converted ckpt 'checkpoint_version' lost"
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
        # ...and the same predicate applied to every bytes/object payload, so
        # the internal `common_state/shard_0_1` blob cannot smuggle the source's
        # unfiltered common state (optimizer entries included) into the output
        # under a key name the check above never inspects.
        assert not hidden_optimizer_keys, (
            f"[{label}] model-weights-only output hides optimizer keys inside a "
            f"bytes payload (predicate {_OPTIMIZER_KEY_RE.pattern!r}): "
            f"{hidden_optimizer_keys[:5]}"
        )
        assert _COMMON_STATE_DCP_KEY not in dst_keys, (
            f"[{label}] model-weights-only output still re-emits the internal "
            f"'{_COMMON_STATE_DCP_KEY}' blob"
        )
        _log(
            rank,
            f"[{label}/{common_state_format}] PASS: {len(recovered)} model weights "
            f"round-tripped, 0 optimizer keys, 0 optimizer keys in bytes payloads",
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


def _build_vpp_source_model_state_dict(num_layers, hidden_size, vocab_size, dtype):
    """Split a GPT state dict into the two VPP model sections the converter must preserve.

    Chunk 0 owns the embedding plus the first half of the layers; chunk 1 owns the second
    half plus the final norm and the output layer. The keys carry the ``model0.`` /
    ``model1.`` section prefix the VPP source layout uses; the bare keys are otherwise
    identical to the single-section fixture.

    Returns ``(sections, chunk0, chunk1)`` where ``sections`` is what gets written to the
    source checkpoint and ``chunk*`` are the per-chunk bare dicts to verify against.
    """
    full = _build_model_state_dict(num_layers, hidden_size, vocab_size, dtype)
    half = num_layers // 2
    chunk0, chunk1 = OrderedDict(), OrderedDict()
    for key, tensor in full.items():
        match = re.search(r'decoder\.layers\.(\d+)\.', key)
        first_chunk = key == 'embedding.word_embeddings.weight' or (
            match is not None and int(match.group(1)) < half
        )
        (chunk0 if first_chunk else chunk1)[key] = tensor

    sections = OrderedDict()
    for prefix, chunk in (('model0.', chunk0), ('model1.', chunk1)):
        for key, tensor in chunk.items():
            sections[f'{prefix}{key}'] = tensor
    return sections, chunk0, chunk1


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


# ---------------------------------------------------------------------------
# VPP model-section key layout (pure; needs neither CUDA nor a process group)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'source_key, expected',
    [
        # Single-section source: the historical `<prefix>.<param>` layout, unchanged.
        (
            'decoder.layers.0.mlp.linear_fc1.weight',
            'model.module.decoder.layers.0.mlp.linear_fc1.weight',
        ),
        # VPP source: the section index must lead the output key.
        (
            'model0.decoder.layers.0.mlp.linear_fc1.weight',
            'model0.module.decoder.layers.0.mlp.linear_fc1.weight',
        ),
        (
            'model1.decoder.layers.4.mlp.linear_fc1.weight',
            'model1.module.decoder.layers.4.mlp.linear_fc1.weight',
        ),
        (
            'model0.embedding.word_embeddings.weight',
            'model0.module.embedding.word_embeddings.weight',
        ),
        # A padded section index is normalized.
        ('model007.x', 'model7.module.x'),
        # `model` as a substring / without a numeric section is not a section.
        ('module0.weight', 'model.module.module0.weight'),
        ('model.0.weight', 'model.module.model.0.weight'),
    ],
)
def test_model_weight_output_key(source_key, expected):
    """The converter must preserve a VPP section index.

    The bug emitted ``model.module.model{i}.<param>`` for a ``model{i}.<param>``
    source, which matches nothing the ``fsdp_dtensor`` loader requests. This pure
    check needs no GPU, so the key layout is covered even where the end-to-end
    conversion cannot run.
    """
    from model_weight_keys import model_weight_output_key

    assert model_weight_output_key(source_key) == expected


@pytest.mark.parametrize(
    'prefix, expected',
    [
        ('model.module', 'model0.module'),
        ('model.foo.bar', 'model0.foo.bar'),
        ('model', 'model0'),
        ('wrapped', 'model0.wrapped'),
        ('wrapped.module', 'model0.wrapped.module'),
    ],
)
def test_sectioned_model_weight_prefix_honours_a_custom_prefix(prefix, expected):
    """``--output-model-weight-prefix`` still selects the wrapper namespace for VPP.

    The section replaces the prefix's leading ``model`` component (the loader
    always wants the section first); a prefix without one is appended after the
    section.
    """
    from model_weight_keys import sectioned_model_weight_prefix

    assert sectioned_model_weight_prefix('model0', prefix) == expected


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


@pytest.mark.usefixtures('_require_cuda_nccl')
def test_vpp_source_keeps_the_section_index():
    """A VPP source (``model0.``/``model1.``) converts to ``model{i}.module.<param>``.

    This is the end-to-end guard for the key-layout fix: the unconditional
    single-section prefix used to produce ``model.module.model{i}.<param>``, which
    matches nothing the VPP ``fsdp_dtensor`` loader requests. Requires CUDA/NCCL
    because ``convert_checkpoint`` builds a CUDA ``DeviceMesh``.
    """
    from checkpoint_inspector import convert_checkpoint

    rank = dist.get_rank()
    case_dir = os.path.join(_shared_root(), 'vpp_sections')
    src_dir = os.path.join(case_dir, 'torch_dist_src', 'iter_0000100')
    dst_dir = os.path.join(case_dir, 'fsdp_dtensor_dst')
    if rank == 0 and os.path.isdir(case_dir):
        shutil.rmtree(case_dir, ignore_errors=True)
    dist.barrier()
    os.makedirs(os.path.dirname(src_dir), exist_ok=True)
    dist.barrier()

    num_layers, hidden_size, vocab_size = 4, 32, 64
    sections, chunk0, chunk1 = _build_vpp_source_model_state_dict(
        num_layers, hidden_size, vocab_size, torch.float32
    )
    _save_source_checkpoint(
        sections,
        _build_common_state(num_layers, hidden_size, vocab_size),
        src_dir,
        common_state_format='current',
        model_prefix='',
    )
    dist.barrier()

    convert_checkpoint(
        src_dir,
        dst_dir,
        False,
        process_group=dist.group.WORLD,
        model_weights_only=True,
    )
    dist.barrier()

    dst_keys = _metadata_keys(dst_dir)
    assert 'model0.module.embedding.word_embeddings.weight' in dst_keys
    assert 'model1.module.output_layer.weight' in dst_keys
    nested = [key for key in dst_keys if key.startswith('model.module.model')]
    assert not nested, sorted(dst_keys)[:5]

    loaded = _load_full_tensors(dst_dir)
    for section, chunk in (('model0', chunk0), ('model1', chunk1)):
        for param, tensor in chunk.items():
            key = f'{section}.module.{param}'
            assert key in loaded, f"[vpp_sections] missing {key}"
            assert torch.equal(loaded[key].to(tensor.dtype), tensor), key

    _log(rank, f"[vpp_sections] PASS: {len(dst_keys)} output keys keep the section index")


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
