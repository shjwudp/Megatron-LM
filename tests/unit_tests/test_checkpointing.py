# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Note: --ckpt-format torch_dist has tests in tests/unit_tests/dist_checkpointing.
import os
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.checkpointing import (
    CheckpointType,
    _build_sharded_state_dict_metadata,
    _load_base_checkpoint,
    get_checkpoint_tracker_filename,
    load_args_from_checkpoint,
    load_checkpoint,
    maybe_save_dataloader_state,
    preprocess_fsdp_dtensor_state_dict,
    read_metadata,
    save_checkpoint,
)
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockModel(MegatronModule):
    """Dummy megatron model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.l = torch.nn.Linear(1, 2)
        torch.nn.init.ones_(self.l.weight)
        torch.nn.init.zeros_(self.l.bias)
        self._called_metadata = []

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


class MockState:
    def __init__(self, state_dict):
        self._state_dict = state_dict
        self.is_stub_optimizer = False
        self._called_metadata = []

        # Optimizers are expected to have this attribute for checkpointing.
        self.param_groups = []

    def state_dict(self, is_loading=False):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def save_parameter_state(self, *args, **kwargs):
        pass

    def load_parameter_state(self, *args, **kwargs):
        pass

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


def test_maybe_save_dataloader_state_uses_explicit_process_groups(tmp_path):
    """Dataloader checkpoints use the supplied module groups and canonical model-parallel path."""
    groups = {
        "tp": SimpleNamespace(rank=0, size=2),
        "pp": SimpleNamespace(rank=0, size=2),
        "dp": SimpleNamespace(rank=3, size=4),
    }
    barriers = []
    saved = []
    iterator = SimpleNamespace(
        iterable=SimpleNamespace(save_state=lambda: {"global_sequence_id": 16})
    )

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank", side_effect=lambda group: group.rank
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size", side_effect=lambda group: group.size
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.distributed.barrier",
            side_effect=lambda group: barriers.append(group),
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.save",
            side_effect=lambda state, path: saved.append((state, path)),
        ),
    ):
        maybe_save_dataloader_state(
            iterator,
            2,
            tmp_path,
            tp_group=groups["tp"],
            pp_group=groups["pp"],
            dp_group=groups["dp"],
        )

    assert barriers == [groups["dp"], groups["dp"]]
    assert saved[0][0] == {"dataloader_state_dict": {"global_sequence_id": 16}}
    assert saved[0][1] == str(
        tmp_path / "iter_0000002" / "mp_rank_00_000" / "train_dataloader_dprank003.pt"
    )


def test_maybe_save_dataloader_state_skips_empty_state_after_barriers(tmp_path):
    """Ranks without dataloader state participate in barriers but do not write a file."""
    group = SimpleNamespace(rank=0, size=1)
    iterator = SimpleNamespace(iterable=SimpleNamespace(save_state=lambda: None))
    barriers = []

    with (
        mock.patch(
            "megatron.training.checkpointing.get_pg_rank",
            side_effect=lambda process_group: process_group.rank,
        ),
        mock.patch(
            "megatron.training.checkpointing.get_pg_size",
            side_effect=lambda process_group: process_group.size,
        ),
        mock.patch(
            "megatron.training.checkpointing.torch.distributed.barrier",
            side_effect=lambda group: barriers.append(group),
        ),
        mock.patch("megatron.training.checkpointing.torch.save") as save,
    ):
        maybe_save_dataloader_state(
            iterator, 2, tmp_path, tp_group=group, pp_group=group, dp_group=group
        )

    assert barriers == [group, group]
    save.assert_not_called()


class MockOptParamScheduler(MockState):
    def __init__(self, state_dict):
        super().__init__(state_dict)
        self.num_steps = state_dict.get("num_steps", 0)
        self.step_calls = []

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.num_steps = state_dict.get("num_steps", self.num_steps)

    def step(self, increment=1):
        self.step_calls.append(increment)


class MockOptimizer(MockState):
    def state_dict(self, is_loading=False):
        state_dict = super().state_dict(is_loading=is_loading).copy()
        state_dict["param_groups"] = [param_group.copy() for param_group in self.param_groups]
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.param_groups = [param_group.copy() for param_group in state_dict["param_groups"]]


@pytest.mark.parametrize(
    ("checkpoint_args", "configured_num_householder", "expected_num_householder"),
    [(SimpleNamespace(gdp_num_householder=5), 3, 5), (SimpleNamespace(), 5, 3)],
)
def test_load_args_restores_gdp_num_householder_from_checkpoint(
    checkpoint_args, configured_num_householder, expected_num_householder
):
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        gdp_num_householder=configured_num_householder,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    state_dict = {"args": checkpoint_args, "iteration": 12}

    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(state_dict, "checkpoint", False, CheckpointType.LEGACY),
    ):
        restored_args, _ = load_args_from_checkpoint(args)

    assert restored_args.gdp_num_householder == expected_num_householder


def create_checkpoint(load_path, ckpt_format):
    """Setup a dummy checkpoint directory."""
    iteration = 123
    ckpt_dir = load_path / "iter_{:07d}".format(iteration)
    tracker_path = get_checkpoint_tracker_filename(load_path)
    with open(tracker_path, "w") as f:
        f.write(str(iteration))

    state_dict = {"args": "dummy", "iteration": iteration}

    if ckpt_format == "torch":
        # Torch checkpoints use a specific directory structure.
        pt_dir = ckpt_dir / "mp_rank_00"
        pt_dir.mkdir(parents=True)
        torch.save(state_dict, pt_dir / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp" and is_torch_min_version("2.4.0"):
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.async_strategy = "mcore"
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.auto_detect_ckpt_format = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.override_opt_param_scheduler = False
    args.swiglu = True
    args.num_experts = 1
    args.verify_integrity = False

    yield args


@pytest.fixture
def create_ckpt_load_args(create_args):
    """Setup dummy args allowing checkpoint load."""
    args = create_args
    args.auto_detect_ckpt_format = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.num_layers = 1
    args.hidden_size = 2
    args.num_attention_heads = 1
    args.add_position_embedding = False
    args.vocab_file = None
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.ckpt_assume_constant_structure = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.ckpt_load_validate_sharding_integrity = True
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.use_megatron_fsdp = False
    args.strict_fsdp_dtensor_load = True
    args.phase_transition_iterations = None

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.mark.parametrize("ckpt_format", ["torch_dcp"])
def test_load_base_checkpoint(
    init_model_parallel, create_ckpt_load_args, ckpt_format, tmp_path_dist_ckpt
):
    """Test _load_base_checkpoint."""

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    # TempNamedDir uses the same directory for all ranks in a multi-GPU setup. Cleanup is handled.
    with TempNamedDir(tmp_path_dist_ckpt / "test_load_base_checkpoint", sync=True) as load_dir:
        create_checkpoint(load_dir, ckpt_format)
        args = create_ckpt_load_args
        args.ckpt_format = ckpt_format

        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=True
        )

    assert state_dict["args"] == "dummy"
    assert state_dict["iteration"] == 123

    expected_ckpt_path = None
    if ckpt_format == "torch":
        expected_ckpt_path = str(load_dir / "iter_0000123" / "mp_rank_00" / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp":
        expected_ckpt_path = str(load_dir / "iter_0000123")

    assert checkpoint_name == expected_ckpt_path
    assert not release

    expected_ckpt_type = None
    if ckpt_format == "torch":
        expected_ckpt_type = CheckpointType.LEGACY
    elif ckpt_format == "torch_dcp":
        expected_ckpt_type = CheckpointType.TORCH_DCP

    assert ckpt_type == expected_ckpt_type


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_save_checkpoint(init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format):
    """Test save_checkpoint."""
    args = create_args
    args.ckpt_format = ckpt_format

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    if ckpt_format == "fsdp_dtensor":
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=True, use_megatron_fsdp=True
            ),
            module=model,
        )
        optimizer = MockState({"state": {}})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_checkpoint", sync=True) as save_dir:
        args.save = save_dir
        args.save_tokenizer_assets = False
        set_args(args)

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        with open(args.save / "latest_checkpointed_iteration.txt", "r") as f:
            assert iteration == int(f.read())

        ckpt_dir = args.save / "iter_0000123"

        expected_ckpt_path = None
        if ckpt_format == "torch":
            expected_ckpt_path = ckpt_dir / "mp_rank_00" / "model_optim_rng.pt"
        elif ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            expected_ckpt_path = ckpt_dir / ".metadata"

        assert os.path.exists(expected_ckpt_path)


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
    """Test load_checkpoint."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    with TempNamedDir(tmp_path_dist_ckpt / "test_load_checkpoint", sync=True) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_floating_point_operations_so_far = 456

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockState({"optimizer": "dummy1"})
        new_opt_param_scheduler = MockState({"opt_param_scheduler": "dummy2"})

        # Load checkpoint
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True
        )

        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far

        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])

        assert new_optimizer.state_dict() == optimizer.state_dict()
        assert new_opt_param_scheduler.state_dict() == opt_param_scheduler.state_dict()


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint_override_opt_param_scheduler(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
    """Test override_opt_param_scheduler behavior during checkpoint load."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = False
    args.use_dist_ckpt = ckpt_format != "torch"
    args.override_opt_param_scheduler = True
    args.lr = 1.0
    args.min_lr = 0.1
    args.decoupled_lr = 0.5
    args.decoupled_min_lr = 0.05
    args.consumed_train_samples = 42

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_load_checkpoint_override_opt_param_scheduler", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockOptimizer({"optimizer": "optimizer_state"})
        optimizer.param_groups = [
            {"is_decoupled_lr": False, "max_lr": -1.0, "min_lr": -1.0},
            {"is_decoupled_lr": True, "max_lr": -1.0, "min_lr": -1.0},
        ]
        opt_param_scheduler = MockOptParamScheduler(
            {"opt_param_scheduler": "scheduler_state", "num_steps": 3}
        )
        num_floating_point_operations_so_far = 456

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockOptimizer({"optimizer": "dummy1"})
        new_optimizer.param_groups = [
            {"is_decoupled_lr": False, "max_lr": -2.0, "min_lr": -2.0},
            {"is_decoupled_lr": True, "max_lr": -2.0, "min_lr": -2.0},
        ]
        new_opt_param_scheduler = MockOptParamScheduler(
            {"opt_param_scheduler": "dummy2", "num_steps": 0}
        )

        # Load checkpoint and verify runtime overrides are restored.
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True
        )
        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far
        assert new_optimizer.param_groups[0]["max_lr"] == args.lr
        assert new_optimizer.param_groups[0]["min_lr"] == args.min_lr
        assert new_optimizer.param_groups[1]["max_lr"] == args.decoupled_lr
        assert new_optimizer.param_groups[1]["min_lr"] == args.decoupled_min_lr
        assert new_opt_param_scheduler.num_steps == args.consumed_train_samples
        assert new_opt_param_scheduler.step_calls[-1] == 0

        # Ensure loading without optimizer/scheduler remains safe.
        loaded_iter_none, loaded_flops_none = load_checkpoint([new_model], None, None, strict=True)
        assert loaded_iter_none == iteration
        assert loaded_flops_none == num_floating_point_operations_so_far


def test_dist_checkpoint_versioning(init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args):
    """Test distributed checkpoint versioning."""
    args = create_ckpt_load_args
    args.ckpt_format = 'torch_dist'
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_dist_checkpoint_versioning", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_fp_ops = 456

        base_metadata = _build_sharded_state_dict_metadata(args)
        first_job_mock_metadata = {**base_metadata, 'metadata_A': 42, 'metadata_B_soon_removed': 43}
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=first_job_mock_metadata,
        ):
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)

        second_job_mock_metadata = {
            **base_metadata,
            'metadata_A': 'changed_default_value',
            'metadata_C_new': {'nested': 'val'},
        }
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=second_job_mock_metadata,
        ):
            # Load checkpoint (into the same model, we don't check load correctness here)
            load_checkpoint([model], optimizer, opt_param_scheduler, strict=True)
            assert optimizer._called_metadata[-1] == first_job_mock_metadata

            # Save the checkpoint again to check if the content metadata for the new checkpoint will be new
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
            assert optimizer._called_metadata[-1] == second_job_mock_metadata

        assert optimizer._called_metadata == model._called_metadata
        assert optimizer._called_metadata == [
            first_job_mock_metadata,
            first_job_mock_metadata,
            second_job_mock_metadata,
        ]


@pytest.mark.parametrize(
    "metadata_content,expected_iter,expected_release",
    [
        ("456", 456, False),  # Normal iteration
        ("release", 0, True),  # Release checkpoint should return iteration=1
        ("123", 123, False),  # Another normal iteration
    ],
)
def test_read_metadata_non_distributed(tmp_path, metadata_content, expected_iter, expected_release):
    """Test read_metadata without torch.distributed initialized."""
    test_dir = tmp_path / "test_read_metadata_non_distributed"
    test_dir.mkdir(parents=True, exist_ok=True)
    tracker_file = test_dir / "latest_checkpointed_iteration.txt"

    with open(tracker_file, "w") as f:
        f.write(metadata_content)

    with mock.patch('torch.distributed.is_initialized', return_value=False):
        max_iter, release = read_metadata(str(tracker_file))

    assert max_iter == expected_iter, f"Expected iteration {expected_iter}, got {max_iter}"
    assert release == expected_release, f"Expected release={expected_release}, got {release}"


def _make_metadata_args(
    use_distributed_optimizer=False,
    use_layer_wise_distributed_optimizer=False,
    ckpt_format='torch_dist',
    dist_ckpt_optim_fully_reshardable=False,
    distrib_optim_fully_reshardable_mem_efficient=False,
):
    args = SimpleNamespace()
    args.use_distributed_optimizer = use_distributed_optimizer
    args.use_layer_wise_distributed_optimizer = use_layer_wise_distributed_optimizer
    args.ckpt_format = ckpt_format
    args.dist_ckpt_optim_fully_reshardable = dist_ckpt_optim_fully_reshardable
    args.distrib_optim_fully_reshardable_mem_efficient = (
        distrib_optim_fully_reshardable_mem_efficient
    )
    return args


class TestBuildShardedStateDictMetadata:
    """``_build_sharded_state_dict_metadata`` must set ``distrib_optim_sharding_type``
    whenever a real :class:`DistributedOptimizer` instance will be used at save
    time -- otherwise the DistOpt path falls through to the deprecated
    ``fully_sharded_model_space`` default whose ``flattened_range`` usage is
    rejected by ``ShardedTensor.validate_metadata_integrity`` post commit
    5ab481cb45.
    """

    DUMMY_GROUP = object()

    def test_distributed_optimizer_sets_dp_reshardable_default(self):
        args = _make_metadata_args(use_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_distributed_optimizer_fully_reshardable_flag(self):
        args = _make_metadata_args(
            use_distributed_optimizer=True, dist_ckpt_optim_fully_reshardable=True
        )
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'
        assert metadata['distrib_optim_fully_reshardable_mem_efficient'] is False

    def test_distributed_optimizer_fsdp_dtensor(self):
        args = _make_metadata_args(use_distributed_optimizer=True, ckpt_format='fsdp_dtensor')
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fsdp_dtensor'

    def test_layer_wise_only_still_sets_sharding_type(self):
        # Arg parser flips ``use_distributed_optimizer`` off when Muon is in
        # use, but the LayerWise + DistOpt split path still has a DistOpt
        # sub-optimizer for non-Muon params, so the metadata is required.
        args = _make_metadata_args(use_layer_wise_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_layer_wise_with_fully_reshardable(self):
        args = _make_metadata_args(
            use_layer_wise_distributed_optimizer=True, dist_ckpt_optim_fully_reshardable=True
        )
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'

    def test_no_distributed_optimizer_no_sharding_type(self):
        args = _make_metadata_args()
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert 'distrib_optim_sharding_type' not in metadata


# ============================================================================
# fsdp_dtensor preprocess: model-section <-> model-chunk pairing (VPP)
# ============================================================================
class TestPreprocessFsdpDtensorStateDictSections:
    """``generate_state_dict`` sections the model per chunk (``model`` for one chunk and
    ``model0``/``model1``/... for virtual pipeline parallelism). The preprocess used to
    index ``state_dict['model']`` unconditionally and raised ``KeyError`` for a VPP state
    dict. These tests pin the replacement: every section is handled with the chunk that
    owns it, the sections are then flattened into one ``model`` section whose layer keys
    carry the global layer index -- so the on-disk convention does not encode the pipeline
    layout -- and the single-section call sequence is unchanged."""

    class _Chunk:
        """Model chunk stub with no layer lists: owns no optimizer key and nothing to rebase."""

        def __init__(self, name):
            self.name = name

        def named_modules(self):
            return []

        def get_parameter(self, name):
            raise AttributeError(name)

    class _NumberedLayer(torch.nn.Module):
        """A stand-in transformer layer: only ``layer_number`` matters here."""

        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number

    class _NumberedChunk(torch.nn.Module):
        """Model chunk stub holding a numbered ``decoder.layers`` list, like a rank's chunk.

        ``first_layer_number`` is the global 1-based number of its first layer, i.e. what a
        real ``TransformerLayer`` gets from ``get_transformer_layer_offset`` at construction.
        """

        def __init__(self, first_layer_number, count):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList(
                [
                    TestPreprocessFsdpDtensorStateDictSections._NumberedLayer(
                        first_layer_number + i
                    )
                    for i in range(count)
                ]
            )

    @staticmethod
    def _args(swiglu=False, num_experts=None):
        return SimpleNamespace(swiglu=swiglu, num_experts=num_experts)

    @staticmethod
    def _patch_handlers(monkeypatch, calls):
        from megatron.training import checkpointing

        def record(name):
            def handler(model_chunk, model_state_dict, optimizer_state_dict):
                calls.append((name, model_chunk, tuple(model_state_dict)))
                return model_state_dict, optimizer_state_dict

            return handler

        monkeypatch.setattr(
            checkpointing,
            'handle_fp8_extra_state_case',
            lambda sd: calls.append(('fp8', None, tuple(sd))),
        )
        monkeypatch.setattr(checkpointing, 'handle_swiglu_in_state_dict', record('swiglu'))
        monkeypatch.setattr(checkpointing, 'handle_mla_down_proj_in_state_dict', record('mla'))
        monkeypatch.setattr(checkpointing, 'handle_mtp_in_state_dict', record('mtp'))
        monkeypatch.setattr(
            checkpointing,
            'handle_experts_in_state_dict',
            lambda sd, n: (calls.append(('experts', None, tuple(sd))), sd)[1],
        )
        monkeypatch.setattr(
            checkpointing,
            'preprocess_state_dict_for_uneven_dtensor',
            lambda sd: calls.append(('uneven', None, tuple(sorted(sd)))),
        )

    def test_vpp_sections_pair_each_chunk_with_its_own_section(self, monkeypatch):
        chunks = [self._Chunk('c0'), self._Chunk('c1')]
        state_dict = {'model0': {'a': 0}, 'model1': {'b': 0}, 'iteration': 1}
        calls = []
        self._patch_handlers(monkeypatch, calls)

        out = preprocess_fsdp_dtensor_state_dict(self._args(), state_dict, chunks)

        # One flat model section, not one section per virtual chunk.
        assert set(out) == {'model', 'iteration'}
        assert out['model'] == {'a': 0, 'b': 0}
        for handler_name in ('mla', 'mtp'):
            assert [chunk for (name, chunk, _keys) in calls if name == handler_name] == chunks
        assert [keys for (name, _chunk, keys) in calls if name == 'fp8'] == [('a',), ('b',)]

    def test_vpp_layer_keys_are_rebased_onto_global_indices(self, monkeypatch):
        """Interleaved VPP: chunk 0 holds global layers 0-1 and chunk 1 holds layers 4-5, so
        the flattened request must ask for 0,1,4,5 rather than 0,0,1,1."""
        chunks = [
            self._NumberedChunk(first_layer_number=1, count=2),
            self._NumberedChunk(first_layer_number=5, count=2),
        ]
        state_dict = {
            'model0': {
                'module.decoder.layers.0.input_layernorm.weight': 'a',
                'module.decoder.layers.1.input_layernorm.weight': 'b',
                'module.embedding.word_embeddings.weight': 'e',
            },
            'model1': {
                'module.decoder.layers.0.input_layernorm.weight': 'c',
                'module.decoder.layers.1.input_layernorm.weight': 'd',
            },
        }
        calls = []
        self._patch_handlers(monkeypatch, calls)

        out = preprocess_fsdp_dtensor_state_dict(self._args(), state_dict, chunks)

        assert set(out) == {'model'}
        assert out['model'] == {
            'module.decoder.layers.0.input_layernorm.weight': 'a',
            'module.decoder.layers.1.input_layernorm.weight': 'b',
            'module.decoder.layers.4.input_layernorm.weight': 'c',
            'module.decoder.layers.5.input_layernorm.weight': 'd',
            'module.embedding.word_embeddings.weight': 'e',
        }
        # The handlers still saw the local indices the module tree has.
        assert [keys for (name, _chunk, keys) in calls if name == 'fp8'] == [
            (
                'module.decoder.layers.0.input_layernorm.weight',
                'module.decoder.layers.1.input_layernorm.weight',
                'module.embedding.word_embeddings.weight',
            ),
            (
                'module.decoder.layers.0.input_layernorm.weight',
                'module.decoder.layers.1.input_layernorm.weight',
            ),
        ]

    def test_sections_that_rebase_onto_the_same_key_raise(self, monkeypatch):
        """A layout whose chunks overlap would let one chunk's weights silently overwrite the
        other's, so the merge must abort instead."""
        chunks = [
            self._NumberedChunk(first_layer_number=1, count=1),
            self._NumberedChunk(first_layer_number=1, count=1),
        ]
        state_dict = {
            'model0': {'module.decoder.layers.0.weight': 'a'},
            'model1': {'module.decoder.layers.0.weight': 'b'},
        }
        calls = []
        self._patch_handlers(monkeypatch, calls)

        with pytest.raises(ValueError, match="disjoint"):
            preprocess_fsdp_dtensor_state_dict(self._args(), state_dict, chunks)

    def test_single_section_keeps_the_historical_call_sequence(self, monkeypatch):
        chunks = [self._Chunk('only')]
        state_dict = {'model': {'a': 0}}
        calls = []
        self._patch_handlers(monkeypatch, calls)

        out = preprocess_fsdp_dtensor_state_dict(self._args(num_experts=4), state_dict, chunks)

        assert out['model'] == {'a': 0}
        assert [name for (name, _chunk, _keys) in calls] == [
            'fp8',
            'mla',
            'experts',
            'mtp',
            'uneven',
        ]
        assert [chunk for (name, chunk, _keys) in calls if name == 'mla'] == chunks

    def test_swiglu_still_runs_per_section_when_enabled(self, monkeypatch):
        chunks = [self._Chunk('c0'), self._Chunk('c1')]
        state_dict = {'model0': {'a': 0}, 'model1': {'b': 0}}
        calls = []
        self._patch_handlers(monkeypatch, calls)

        preprocess_fsdp_dtensor_state_dict(self._args(swiglu=True), state_dict, chunks)

        assert [chunk for (name, chunk, _keys) in calls if name == 'swiglu'] == chunks

    def test_vpp_optimizer_entries_are_not_dropped(self, monkeypatch):
        """The optimizer is a single unsectioned entry; entries owned by no chunk (here,
        because the stub chunks expose no parameters) must survive untouched rather than
        be silently dropped when the per-chunk slices are merged back."""
        chunks = [self._Chunk('c0'), self._Chunk('c1')]
        optimizer = {'state': {'p': {'exp_avg': 1}}, 'param_to_group_meta': {'p': {}}}
        state_dict = {'model0': {'a': 0}, 'model1': {'b': 0}, 'optimizer': optimizer}
        calls = []
        self._patch_handlers(monkeypatch, calls)

        out = preprocess_fsdp_dtensor_state_dict(self._args(), state_dict, chunks)

        assert out['optimizer']['state'] == {'p': {'exp_avg': 1}}
        assert out['optimizer']['param_to_group_meta'] == {'p': {}}
