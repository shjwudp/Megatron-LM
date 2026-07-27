# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused CPU tests for M-FSDP CUDA Graph recording."""

import builtins
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import hooks as mfsdp_hooks
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import TracePoolAllocator
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.cuda_graph_runner import (
    CudaGraphRunner,
    _capture_module_topology,
    _clone_capture_sample,
    _infer_activation_recompute_regions,
    _make_module_topology_preflight,
    _normalize_forward_call,
    _renew_fsdp_compute_parameter_leaves,
    _validate_activation_recompute_lifetime,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks import (
    _cuda_graph_replay_phase,
    _current_graph_task_id,
    _is_activation_recompute,
    _maybe_capture_cuda_graphs,
    _output_supports_backward,
    _recover_stale_root_backward,
    _should_recover_stale_root_backward,
)


def test_normalize_forward_call_preserves_variadic_arguments():
    """Keep variadic arguments at their original call level."""

    class VariadicModule(torch.nn.Module):
        """Module with positional, variadic, and keyword inputs."""

        def forward(self, hidden_states, /, metadata=None, *extra_states, **kwargs):
            """Return the input tensor.

            :param hidden_states: Input tensor.
            :type hidden_states: torch.Tensor
            :param metadata: Optional metadata.
            :type metadata: Any
            :param extra_states: Additional positional tensors.
            :type extra_states: tuple
            :param kwargs: Additional keyword values.
            :type kwargs: dict
            :return: Input tensor.
            :rtype: torch.Tensor
            """
            del metadata, extra_states, kwargs
            return hidden_states

    module = VariadicModule()
    hidden = torch.ones(2)
    extra = torch.full((2,), 2.0)
    args, kwargs = _normalize_forward_call(
        module, (hidden, None, extra), {"rotary": (torch.ones(2),)}
    )
    assert args == (hidden, None, extra)
    assert tuple(kwargs) == ("rotary",)


def test_recompute_phase_uses_mfsdp_backward_context():
    """Select RF for a module forward executed inside M-FSDP backward."""
    context = SimpleNamespace(
        backward_phase=False,
        backward_module=None,
        cuda_graph_activation_recompute=True,
        forward_phase=False,
    )
    param_group = SimpleNamespace(params=[torch.nn.Parameter(torch.ones(1))])
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        _fsdp_param_groups=(param_group,),
        cuda_graph_compatible=True,
        training=True,
    )
    assert not _is_activation_recompute(module)
    assert _cuda_graph_replay_phase(module) == "forward"
    with torch.no_grad():
        assert not _is_activation_recompute(module)
        assert _cuda_graph_replay_phase(module) == "inference"
        context.forward_phase = True
        assert _cuda_graph_replay_phase(module) == "inference"
        context.forward_phase = False
    context.backward_phase = True
    context.backward_module = id(module)
    assert _is_activation_recompute(module)
    assert _cuda_graph_replay_phase(module) == "recompute"
    module.training = False
    assert _cuda_graph_replay_phase(module) == "inference"


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("backward_phase", [False, True])
@pytest.mark.parametrize("activation_recompute", [False, True])
def test_cuda_graph_replay_phase_truth_table(
    training, grad_enabled, backward_phase, activation_recompute
):
    """Classify every training, grad, backward, and recompute combination.

    :param training: Whether the module is in training mode.
    :type training: bool
    :param grad_enabled: Whether autograd recording is enabled.
    :type grad_enabled: bool
    :param backward_phase: Whether M-FSDP is executing backward.
    :type backward_phase: bool
    :param activation_recompute: Whether three-graph recompute is enabled.
    :type activation_recompute: bool
    """
    module = SimpleNamespace(
        _fsdp_state=SimpleNamespace(enable_cuda_graph=False),
        cuda_graph_compatible=True,
        training=training,
    )
    module._fsdp_root_context = SimpleNamespace(
        backward_module=id(module) if backward_phase else None,
        backward_phase=backward_phase,
        cuda_graph_activation_recompute=activation_recompute,
        cuda_graph_runner=None,
        forward_phase=not backward_phase,
    )
    expected = (
        "inference"
        if not training or not grad_enabled
        else "recompute" if backward_phase else "forward"
    )

    with torch.set_grad_enabled(grad_enabled):
        assert _cuda_graph_replay_phase(module) == expected


def test_grad_enabled_side_forward_is_rejected_during_backward():
    """Reject a side forward that is not the active recompute module."""
    context = SimpleNamespace(
        backward_phase=True,
        backward_module=None,
        cuda_graph_activation_recompute=True,
        forward_phase=False,
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        _fsdp_param_groups=(),
        cuda_graph_compatible=True,
        training=True,
    )

    with pytest.raises(RuntimeError, match="outside checkpoint recomputation"):
        _cuda_graph_replay_phase(module)


def test_plain_mfsdp_checkpoint_forward_remains_recompute():
    """Do not apply three-graph side-forward guards to ordinary M-FSDP."""
    context = SimpleNamespace(
        backward_phase=True,
        backward_module=None,
        cuda_graph_activation_recompute=False,
        forward_phase=False,
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        cuda_graph_compatible=True,
        training=True,
    )
    context.backward_module = id(module)

    assert _cuda_graph_replay_phase(module) == "recompute"


def test_plain_mfsdp_pipeline_forward_remains_forward():
    """Keep an interleaved PP forward out of recompute dispatch."""
    context = SimpleNamespace(
        backward_phase=True,
        backward_module=None,
        cuda_graph_activation_recompute=False,
        forward_phase=False,
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        cuda_graph_compatible=True,
        training=True,
    )

    assert _cuda_graph_replay_phase(module) == "forward"


def test_te_checkpoint_recompute_uses_active_graph_task(monkeypatch):
    """Recognize TE checkpoint recompute without the MCore marker.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    context = SimpleNamespace(
        backward_phase=True,
        backward_module=None,
        cuda_graph_activation_recompute=False,
        forward_phase=False,
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        cuda_graph_compatible=True,
        training=True,
    )
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: 7)

    assert _cuda_graph_replay_phase(module) == "recompute"


def test_recompute_detection_tolerates_missing_mcore(monkeypatch):
    """Keep standalone megatron-fsdp usable without Megatron checkpoint helpers."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Raise only for the optional Megatron checkpoint helper."""
        if name == "megatron.core.tensor_parallel.random":
            raise ImportError("standalone package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    context = SimpleNamespace(
        backward_phase=True, backward_module=None, cuda_graph_activation_recompute=True
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        cuda_graph_compatible=True,
        training=True,
    )

    assert not _is_activation_recompute(module)


def test_graph_task_id_falls_back_when_torch_api_is_unavailable(monkeypatch):
    """Return the idle sentinel when PyTorch lacks the private task API.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    monkeypatch.setattr(torch._C, "_current_graph_task_id", None, raising=False)

    assert _current_graph_task_id() == -1


def test_non_graphed_checkpoint_member_remains_recompute(monkeypatch):
    """Allow an incompatible module inside a larger checkpoint region."""
    context = SimpleNamespace(
        backward_phase=True, backward_module=None, cuda_graph_activation_recompute=True
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        cuda_graph_compatible=False,
        training=True,
    )
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: 1)

    assert _is_activation_recompute(module)


def test_pending_forward_does_not_authorize_side_recompute(monkeypatch):
    """Reject a side forward until backward preparation arms RF dispatch."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    module._fsdp_state = SimpleNamespace(enable_cuda_graph=True)
    module.cuda_graph_compatible = True
    module.training = True
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    runner.record_module(module, (torch.ones(2, 4),), {})
    context = SimpleNamespace(
        backward_phase=True,
        backward_module=id(module),
        cuda_graph_activation_recompute=True,
        cuda_graph_runner=runner,
        forward_phase=False,
    )
    module._fsdp_root_context = context

    with pytest.raises(RuntimeError, match="outside checkpoint recomputation"):
        _cuda_graph_replay_phase(module)
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: 1)
    assert _cuda_graph_replay_phase(module) == "recompute"
    runner.record_module_backward(module)
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: -1)
    with pytest.raises(RuntimeError, match="outside checkpoint recomputation"):
        _cuda_graph_replay_phase(module)


def test_frozen_no_grad_module_uses_inference_phase():
    """Do not create pending backward state for a frozen teacher forward."""
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    context = SimpleNamespace(
        backward_phase=False,
        backward_module=None,
        cuda_graph_activation_recompute=True,
        forward_phase=True,
    )
    module = SimpleNamespace(
        _fsdp_root_context=context,
        _fsdp_param_groups=(SimpleNamespace(params=[frozen]),),
        training=True,
    )

    with torch.no_grad():
        assert _cuda_graph_replay_phase(module) == "inference"


def test_stale_backward_recovery_clears_module_state():
    """Reset state left by a backward exception before retrying forward."""
    release_calls = []
    child = SimpleNamespace(
        _fsdp_cg_pending_backwards=1,
        _fsdp_pre_backward_done=True,
        _fsdp_post_backward_hook_seen=True,
        post_backward_issued=True,
    )
    context = SimpleNamespace(
        backward_done_modules={1},
        backward_module=1,
        backward_phase=True,
        cuda_graph_runner=SimpleNamespace(
            release_pending=lambda: release_calls.append(True) or True
        ),
        forward_order=[child],
        forward_phase=False,
    )
    root = SimpleNamespace(
        _fsdp_root_context=context, _fsdp_state=SimpleNamespace(_post_backward_callback_queued=True)
    )

    _recover_stale_root_backward(root)

    assert release_calls == [True]
    assert not context.backward_phase
    assert context.backward_module is None
    assert not context.backward_done_modules
    assert not root._fsdp_state._post_backward_callback_queued
    assert child._fsdp_cg_pending_backwards == 0
    assert not child._fsdp_pre_backward_done
    assert not child._fsdp_post_backward_hook_seen
    assert not child.post_backward_issued


def test_multi_lane_forward_does_not_auto_release_live_backward(monkeypatch):
    """Keep live PP invocations when a new root forward starts.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(backward_phase=True, cuda_graph_max_pending_forwards=2),
        _fsdp_state=SimpleNamespace(_is_root=True),
        training=True,
    )
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: -1)

    assert not _should_recover_stale_root_backward(module)
    module._fsdp_root_context.cuda_graph_max_pending_forwards = 1
    assert _should_recover_stale_root_backward(module)


def test_detached_training_output_releases_pending_state(monkeypatch):
    """Reject a detached training output without leaking pending state.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    released = []
    module = torch.nn.Linear(2, 2)
    module._fsdp_root_context = SimpleNamespace(
        backward_phase=False,
        backward_module=None,
        cuda_graph_active=False,
        cuda_graph_runner=SimpleNamespace(release_pending=lambda: released.append(True) or True),
    )
    module._fsdp_state = SimpleNamespace(_is_root=False, enable_cuda_graph=True)
    module._fsdp_cg_pending_backwards = 1
    module._fsdp_forward_replay_phase_stack = [("forward", True)]
    module.cuda_graph_compatible = True
    module.reshard = lambda: None
    monkeypatch.setattr(mfsdp_hooks, "FSDPModule", torch.nn.Module)

    detached = torch.ones(2, 2)
    assert not _output_supports_backward(detached)
    with pytest.raises(RuntimeError, match="produced no output"):
        mfsdp_hooks.mfsdp_post_forward_hook(module, detached)

    assert module._fsdp_cg_pending_backwards == 0
    assert released == [True]


def test_checkpoint_early_stop_keeps_pending_state(monkeypatch):
    """Keep the invocation token when non-reentrant recompute stops early."""
    released = []
    module = torch.nn.Linear(2, 2)
    module._fsdp_root_context = SimpleNamespace(
        backward_phase=True,
        backward_module=None,
        cuda_graph_active=False,
        cuda_graph_runner=SimpleNamespace(release_pending=lambda: released.append(True) or True),
    )
    module._fsdp_state = SimpleNamespace(_is_root=False, enable_cuda_graph=True)
    module._fsdp_cg_pending_backwards = 1
    module._fsdp_forward_replay_phase_stack = [("recompute", False)]
    module.cuda_graph_compatible = True
    module.reshard = lambda: None
    monkeypatch.setattr(mfsdp_hooks, "FSDPModule", torch.nn.Module)

    try:
        raise mfsdp_hooks._StopRecomputationError
    except mfsdp_hooks._StopRecomputationError:
        mfsdp_hooks.mfsdp_post_forward_hook(module, None)

    assert module._fsdp_cg_pending_backwards == 1
    assert released == []


def test_fine_grained_output_hook_does_not_consume_root_token():
    """Leave the root invocation token for the direct root output hook."""
    token_calls = []
    target = torch.nn.Linear(2, 2)
    target._fsdp_root_context = SimpleNamespace(
        backward_phase=False,
        cuda_graph_active=False,
        cuda_graph_runner=SimpleNamespace(
            backward_invocation_token=lambda module: token_calls.append(module)
        ),
    )
    child = torch.nn.Linear(2, 2)
    child._fsdp_parent_module = weakref.ref(target)
    handle = mfsdp_hooks._create_custom_backward_hook(child, lambda *args: None)
    try:
        child(torch.ones(2, 2, requires_grad=True))
    finally:
        handle.remove()

    assert token_calls == []


def test_post_forward_rolls_back_only_an_incremented_pending_count(monkeypatch):
    """Keep an earlier microbatch pending when pre-forward fails before increment."""

    class FakeFSDPModule:
        """Minimal direct module accepted by the post-forward hook."""

        def __init__(self, incremented):
            """Create one failed-forward frame.

            :param incremented: Whether pre-forward incremented the pending count.
            :type incremented: bool
            """
            self._fsdp_root_context = SimpleNamespace(
                backward_module=None,
                backward_phase=False,
                cuda_graph_active=False,
                cuda_graph_runner=None,
            )
            self._fsdp_state = SimpleNamespace(_is_root=False, enable_cuda_graph=True)
            self._fsdp_forward_replay_phase_stack = [("forward", incremented)]
            self._fsdp_cg_installed = False
            self._fsdp_cg_pending_backwards = 2
            self.cuda_graph_compatible = True
            self.resharded = False

        def reshard(self):
            """Record post-forward cleanup."""
            self.resharded = True

    monkeypatch.setattr(mfsdp_hooks, "FSDPModule", FakeFSDPModule)
    not_incremented = FakeFSDPModule(False)
    incremented = FakeFSDPModule(True)

    for module in (not_incremented, incremented):
        try:
            raise RuntimeError("pre-forward failed")
        except RuntimeError:
            mfsdp_hooks.mfsdp_post_forward_hook(module, None)

    assert not_incremented._fsdp_cg_pending_backwards == 2
    assert incremented._fsdp_cg_pending_backwards == 1
    assert not_incremented.resharded and incremented.resharded


def test_capture_is_deferred_until_next_root_forward():
    """Run requested graph capture only from the later safe point."""
    calls = []
    allocator = TracePoolAllocator()
    allocator._phase = "optimized"

    class Runner:
        """Minimal graph runner used by the deferred-capture test."""

        captured = False

        def capture_and_install(self, root, capture_stream):
            """Record capture and mark the runner installed.

            :param root: Root module passed to capture.
            :type root: object
            :param capture_stream: Capture stream.
            :type capture_stream: object
            """
            calls.append((root, capture_stream))
            self.captured = True

    runner = Runner()
    stream = object()
    context = SimpleNamespace(
        bucket_allocator=allocator,
        cuda_graph_capture_pending=False,
        cuda_graph_runner=runner,
        cuda_graph_stream=stream,
    )
    root = SimpleNamespace(training=True)

    _maybe_capture_cuda_graphs(context, root, defer=True)
    assert context.cuda_graph_capture_pending
    assert not calls

    _maybe_capture_cuda_graphs(context, root)
    assert not context.cuda_graph_capture_pending
    assert calls == [(root, stream)]


def test_deferred_capture_skips_eval_and_retries_training():
    """Keep capture pending across an evaluation forward."""
    calls = []
    allocator = TracePoolAllocator()
    allocator._phase = "optimized"

    class Runner:
        """Minimal graph runner with explicit capture state."""

        captured = False

        def capture_and_install(self, root, capture_stream):
            """Record one successful capture.

            :param root: Root module passed to capture.
            :type root: object
            :param capture_stream: Capture stream.
            :type capture_stream: object
            """
            calls.append((root, capture_stream))
            self.captured = True

    runner = Runner()
    context = SimpleNamespace(
        bucket_allocator=allocator,
        cuda_graph_capture_pending=True,
        cuda_graph_runner=runner,
        cuda_graph_stream=object(),
    )
    root = SimpleNamespace(training=False)

    _maybe_capture_cuda_graphs(context, root)
    assert context.cuda_graph_capture_pending
    assert not calls

    root.training = True
    _maybe_capture_cuda_graphs(context, root)
    assert not context.cuda_graph_capture_pending
    assert len(calls) == 1


def test_capture_sample_uses_recompute_requires_grad_surface():
    """Build the static input from RF rather than original-forward metadata."""
    forward_input = {"hidden": torch.ones(2), "mask": torch.ones(2, dtype=torch.bool)}
    recompute_surface = {"hidden": True, "mask": False}
    cloned = _clone_capture_sample(forward_input, recompute_surface)

    assert cloned["hidden"].requires_grad
    assert not cloned["mask"].requires_grad
    assert cloned["hidden"].data_ptr() != forward_input["hidden"].data_ptr()


def test_module_topology_preflight_uses_cached_owners():
    """Reject slot replacement without a recursive replay-time walk."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module[0].register_buffer("scale", torch.ones(1))
    preflight = _make_module_topology_preflight(_capture_module_topology(module))

    module[0].register_buffer("offset", torch.zeros(1))
    with pytest.raises(RuntimeError, match="registered buffer topology changed"):
        preflight()
    del module[0]._buffers["offset"]

    def fail_recursive_walk(*args, **kwargs):
        """Reject an unexpected recursive walk."""
        del args, kwargs
        raise AssertionError("replay walked named_modules")

    module.named_modules = fail_recursive_walk
    preflight()


def test_activation_recompute_validates_serial_regions():
    """Accept serial regions and reject a recompute-order mismatch."""
    _validate_activation_recompute_lifetime(
        [
            ("forward", 0, 0),
            ("forward", 1, 0),
            ("forward", 2, 1),
            ("recompute", 2, 1),
            ("backward", 2, 1),
            ("recompute", 0, 0),
            ("recompute", 1, 0),
            ("backward", 1, 0),
            ("backward", 0, 0),
        ],
        module_regions=(0, 0, 1),
    )
    with pytest.raises(RuntimeError, match="RF in forward order"):
        _validate_activation_recompute_lifetime(
            [
                ("forward", 0, 0),
                ("forward", 1, 0),
                ("recompute", 1, 0),
                ("recompute", 0, 0),
                ("backward", 1, 0),
                ("backward", 0, 0),
            ],
            module_regions=(0, 0),
        )


def test_runner_infers_recompute_regions_from_execution_order():
    """Infer region membership without checkpoint-owned markers."""
    regions, events = _infer_activation_recompute_regions(
        [
            ("forward", 0, -1),
            ("forward", 1, -1),
            ("forward", 2, -1),
            ("recompute", 2, -1),
            ("backward", 2, -1),
            ("recompute", 0, -1),
            ("recompute", 1, -1),
            ("backward", 1, -1),
            ("backward", 0, -1),
        ],
        3,
        require_reverse_regions=True,
    )
    assert regions == (0, 0, 1)
    assert events == (("forward", 0), ("forward", 1), ("backward", 1), ("backward", 0))


def test_runner_tracks_reverse_microbatch_lanes_without_checkpoint_tokens():
    """Use output-owned invocation tokens to record reverse backward order."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner.record_module(module, (torch.ones(2, 4),), {})
    first_token = runner.backward_invocation_token(module)
    runner.record_module(module, (torch.full((2, 4), 2.0),), {})
    second_token = runner.backward_invocation_token(module)

    runner.select_backward_invocation(module, second_token)
    runner.record_module_recompute(module)
    runner.record_module_backward(module)
    assert runner.complete_module_backward(module)

    runner.select_backward_invocation(module, first_token)
    runner.record_module_recompute(module)
    runner.record_module_backward(module)
    assert runner.complete_module_backward(module)

    plan = runner._build_ordered_capture_plan()
    assert [event[2] for event in plan.replay_events] == [0, 1, 1, 1, 0, 0]


def test_runner_rejects_forward_lane_wrap_before_backward():
    """Reject a third forward while both configured lanes remain pending."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)

    for value in (1.0, 2.0):
        runner.record_module(module, (torch.full((2, 4), value),), {})
        runner.backward_invocation_token(module)

    with pytest.raises(RuntimeError, match=r"release_pending\(\)"):
        runner.preflight_record_module(module, "forward")
    assert len(runner._ordered_invocations) == 2


def test_runner_reuses_lane_released_by_non_fifo_backward():
    """Reuse the lane whose backward finished before an older lane."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)

    runner.record_module(module, (torch.ones(2, 4),), {})
    first_token = runner.backward_invocation_token(module)
    runner.record_module(module, (torch.full((2, 4), 2.0),), {})
    second_token = runner.backward_invocation_token(module)
    runner.select_backward_invocation(module, second_token)
    runner.record_module_recompute(module)
    runner.record_module_backward(module)
    assert runner.complete_module_backward(module)

    runner.record_module(module, (torch.full((2, 4), 3.0),), {})

    assert runner._ordered_invocations[-1].lane_index == 1
    assert first_token is not None


def test_multi_lane_recompute_requires_output_invocation_token():
    """Reject RF when backward did not select an instrumented output lane."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner.record_module(module, (torch.ones(2, 4),), {})
    runner.backward_invocation_token(module)

    with pytest.raises(RuntimeError, match="invocation-specific output backward hook"):
        runner.record_module_recompute(module)


def test_runner_deduplicates_multi_lane_recompute_hooks():
    """Record one RF event when fine-grained hooks repeat."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner.record_module(module, (torch.ones(2, 4),), {})
    token = runner.backward_invocation_token(module)
    runner.select_backward_invocation(module, token)

    runner.record_module_recompute(module)
    runner.record_module_recompute(module, ("child", "arguments"), {})

    assert [event[0] for event in runner._ordered_lifetime_events].count("recompute") == 1


def test_runner_deduplicates_single_lane_before_normalizing_hook_args():
    """Ignore a repeated fine-grained RF hook before binding child arguments."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    runner.record_module(module, (torch.ones(2, 4),), {})

    runner.record_module_recompute(module, (torch.ones(2, 4, requires_grad=True),), {})
    runner.record_module_recompute(module, ("child", "arguments"), {})

    assert [event[0] for event in runner._lifetime_events].count("recompute") == 1


def test_runner_identifies_only_selected_recompute_invocation():
    """Use the selected output token to distinguish RF from a side forward."""
    module = torch.nn.Linear(4, 4)
    side_module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    side_module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner.record_module(module, (torch.ones(2, 4),), {})
    token = runner.backward_invocation_token(module)
    runner.select_backward_invocation(module, token)

    assert runner.expects_module_recompute(module)
    assert not runner.expects_module_recompute(side_module)
    runner.record_module_recompute(module)
    assert not runner.expects_module_recompute(module)


def test_runner_preserves_shared_module_invocations_per_forward_lane():
    """Queue repeated module calls without collapsing their root-output token."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    module._fsdp_state = SimpleNamespace(_is_root=False)
    root = SimpleNamespace(_fsdp_state=SimpleNamespace(_is_root=True))
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    tokens = []

    for value in (1.0, 2.0):
        runner.begin_forward_scope()
        runner.record_module(module, (torch.full((2, 4), value),), {})
        first = runner.backward_invocation_token(module)
        runner.record_module(module, (torch.full((2, 4), value + 0.5),), {})
        second = runner.backward_invocation_token(module)
        root_token = runner.backward_invocation_token(root)
        assert len(root_token.invocations) == 2
        tokens.append((first, second, root_token))

    for first, second, root_token in reversed(tokens):
        runner.select_backward_invocation(root, root_token)
        runner.select_backward_invocation(module, second)
        runner.record_module_recompute(module)
        runner.record_module_backward(module)
        assert runner.complete_module_backward(module)
        runner.select_backward_invocation(module, first)
        runner.record_module_recompute(module)
        runner.record_module_backward(module)
        assert runner.complete_module_backward(module)

    assert [invocation.lane_index for invocation in runner._ordered_invocations] == [0, 0, 1, 1]
    plan = runner._build_ordered_capture_plan()
    assert plan.order_slots == (0, 0, 1, 1, 1, 1, 0, 0)


def test_runner_rejects_multi_lane_backward_before_recompute():
    """Do not complete an invocation before its RF event."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner.record_module(module, (torch.ones(2, 4),), {})
    token = runner.backward_invocation_token(module)
    runner.select_backward_invocation(module, token)
    runner.record_module_backward(module)

    with pytest.raises(RuntimeError, match="wrap the graph-enabled module"):
        runner.complete_module_backward(module, strict=True)

    runner.record_module_recompute(module)
    assert runner.complete_module_backward(module, strict=True)


def test_runner_diagnoses_missing_activation_checkpoint():
    """Explain activation-recompute configuration without checkpoint RF."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    runner.record_module(module, (torch.ones(2, 4),), {})
    runner.record_module_backward(module)

    with pytest.raises(RuntimeError, match="disable cuda_graph_activation_recompute"):
        runner.complete_module_backward(module, strict=True)


def test_runner_completes_plain_cuda_graph_backward():
    """Keep ordinary retained CUDA Graph backward independent of RF state."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=False)
    runner.record_module(module, (torch.ones(2, 4),), {})
    runner.record_module_backward(module)

    assert runner.complete_module_backward(module, strict=True)
    assert not runner.complete_module_backward(module)


def test_runner_encodes_interleaved_microbatch_lanes_in_order():
    """Preserve a recorded 1F1B lane schedule in the custom order."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)

    for value in (1.0, 2.0):
        runner.record_module(module, (torch.full((2, 4), value),), {})
        token = runner.backward_invocation_token(module)
        runner.select_backward_invocation(module, token)
        runner.record_module_recompute(module)
        runner.record_module_backward(module)
        assert runner.complete_module_backward(module)

    plan = runner._build_ordered_capture_plan()
    assert plan.order == (1, -1, 1, -1)
    assert plan.order_slots == (0, 0, 1, 1)
    assert [event[2] for event in plan.replay_events] == [0, 0, 0, 1, 1, 1]


def test_root_output_token_rejects_released_forward():
    """Reject backward from a root output released before capture."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    module._fsdp_cg_pending_backwards = 1
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    runner.record_module(module, (torch.ones(2, 4),), {})
    token = runner.backward_invocation_token(SimpleNamespace())

    assert runner.release_pending()
    with pytest.raises(RuntimeError, match="released or superseded"):
        runner.select_backward_invocation(SimpleNamespace(), token)


def test_reset_invalidates_existing_root_output_token():
    """Reject an old output even when reset starts another recording epoch."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    runner.record_module(module, (torch.ones(2, 4),), {})
    token = runner.backward_invocation_token(module)

    runner.reset()

    with pytest.raises(RuntimeError, match="released or superseded"):
        runner.select_backward_invocation(module, token)


def test_multi_lane_inference_has_explicit_error():
    """Reject inference before reporting an unrelated replay-order mismatch."""
    module = torch.nn.Linear(4, 4)
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner._captured = True
    runner._ordered_module_wrappers = {id(module): [(0, object())]}
    runner._ordered_replay_events = [("forward", id(module), 0)]
    runner._active_backward_invocations[id(module)] = 0

    with pytest.raises(RuntimeError, match="idle schedule"):
        runner.prepare_module_replay(module, "inference")


def test_multi_lane_recompute_requires_token_after_capture():
    """Reject captured RF before a root output selects its invocation."""
    module = torch.nn.Linear(4, 4)
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner._captured = True
    runner._ordered_module_wrappers = {id(module): [(0, object())]}
    runner._ordered_replay_events = [("recompute", id(module), 0)]
    runner._ordered_slot_lanes = {0: 0}

    with pytest.raises(RuntimeError, match="invocation-specific output backward hook"):
        runner.prepare_module_replay(module, "recompute")


def test_multi_lane_idle_inference_does_not_advance_schedule():
    """Run idle inference without consuming an ordered training event."""
    phases = []

    def wrapper(value):
        """Return a distinct inference result.

        :param value: Runtime input.
        :type value: torch.Tensor
        :return: Shifted output.
        :rtype: torch.Tensor
        """
        return value + 1

    wrapper._cuda_graph_set_replay_phase = phases.append
    wrapper._cuda_graph_preflight = lambda: None
    module = torch.nn.Linear(4, 4)
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True, max_pending_forwards=2)
    runner._captured = True
    runner._ordered_module_wrappers = {id(module): [(0, wrapper)]}
    runner._ordered_replay_events = [("forward", id(module), 0), ("forward", id(module), 0)]
    runner._ordered_replay_cursor = 1
    runner._ordered_slot_lanes = {0: 0}
    _ = runner._queued_backward_invocations[id(module)]

    runner.prepare_module_replay(module, "inference")
    runner._ordered_preflight_replay(module)
    output = runner._ordered_dispatch(module, torch.ones(1))

    torch.testing.assert_close(output, torch.full((1,), 2.0))
    assert phases == ["inference"]
    assert runner._ordered_replay_cursor == 1


def test_capture_renews_only_internal_compute_leaves():
    """Keep optimizer-facing registered parameters stable during leaf renewal."""
    module = torch.nn.Linear(2, 2)
    compute_params = list(module.parameters())
    names = ["weight", "bias"]
    param_idx = {parameter: index for index, parameter in enumerate(compute_params)}
    buffers = [SimpleNamespace(params=compute_params, param_idx=param_idx) for _ in range(2)]
    dist_params = [torch.nn.Parameter(parameter.detach().clone()) for parameter in compute_params]
    module.weight, module.bias = dist_params
    param_group = SimpleNamespace(
        params=compute_params,
        dist_params=dist_params,
        param_idx=param_idx,
        model_weight_buffer=buffers[0],
        transpose_weight_buffer=None,
        main_weight_buffer=None,
        main_grad_buffer=buffers[1],
    )
    module._named_param_groups = [(names, param_group)]
    module._init_param_main_grad_func = lambda: None

    _renew_fsdp_compute_parameter_leaves((module,))

    assert list(module.parameters()) == dist_params
    assert all(new is not old for new, old in zip(param_group.params, compute_params))
    assert [new.data_ptr() for new in param_group.params] == [
        old.data_ptr() for old in compute_params
    ]
    assert buffers[0].params is param_group.params
    assert buffers[1].params is param_group.params
