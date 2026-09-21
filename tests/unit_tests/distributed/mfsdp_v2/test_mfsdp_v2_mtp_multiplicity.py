# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient completion for a shared parameter under combined 1F1B.

The combined/fine-grained 1F1B backward is one autograd GraphTask per schedule
node over detached node inputs, so a parameter's hooks fire once per
``(parameter, consuming node)`` pair rather than once per iteration. The shared
embedding is the visible case: the chunk's PreProcessNode and the MTP
pre-dispatch node each consume it, so one iteration yields two contributions for
one parameter and a fire count cannot decide when the window is complete.

``MultiplicityReadiness`` replaces that count with a per-parameter declaration --
a shortfall holds the window open, a surplus raises -- and
``register_combined_1f1b_hooks`` installs that declaration from
``_unit_grad_multiplicity``. Both layers are exercised here: the accounting
contract on its own, and the declaration as it lands on the real ``FsdpParameter``
objects ``fully_shard`` produces.

The stack-dependent tests are gated on a host precondition evaluated *before* the
import it guards, never on ``except ImportError`` -- an import failure on a host
that satisfies the precondition must fail loudly rather than dissolve into a
skip. The imports themselves live in fixtures: a module-level import of the
Megatron stack would turn a host that cannot import it into a collection error
instead of a skip.
"""

import importlib
import os
from types import SimpleNamespace

import pytest

_EMBEDDING_HINT = "word_embeddings"
_OUTPUT_HINT = "output_layer"


def _host_can_import_stack() -> bool:
    """Return whether this host can import the Megatron stack at all."""
    try:
        importlib.import_module("megatron.core.distributed.fsdp.src.megatron_fsdp.experimental")
    except Exception:  # any failure here is the precondition, not the test
        return False
    return True


# ``distributed_setup`` reads torchrun's rank state, so the declaration layer
# additionally needs to be running under ``torch.distributed.run``.
_HOST_CAN_IMPORT_STACK = _host_can_import_stack()
_UNDER_TORCHRUN = "RANK" in os.environ and "WORLD_SIZE" in os.environ

_needs_stack = pytest.mark.skipif(
    not _HOST_CAN_IMPORT_STACK,
    reason="host precondition unmet: this torch cannot import the Megatron stack, "
    "which is unrelated to the gradient multiplicity under test.",
)
_needs_distributed = pytest.mark.skipif(
    not (_HOST_CAN_IMPORT_STACK and _UNDER_TORCHRUN),
    reason="host precondition unmet: needs both an importable Megatron stack and "
    "torchrun rank state.",
)


@pytest.fixture
def readiness_cls():
    """The readiness tracker under test."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown import (
        MultiplicityReadiness,
    )

    return MultiplicityReadiness


@_needs_stack
class TestSharedParameterAcrossGraphTasks:
    """One parameter, two consuming schedule nodes, one completion window."""

    def test_a_declared_second_contribution_keeps_the_window_open(self, readiness_cls):
        """The positive case: the window does not close after the first node."""
        embedding, norm = ("embedding.weight",), ("norm.weight",)
        readiness = readiness_cls({embedding: 2, norm: 1})
        readiness.mark(norm)
        readiness.mark(embedding)  # PreProcessNode embedding lookup
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({embedding})
        readiness.mark(embedding)  # MTP pre-dispatch node
        assert readiness.is_complete()

    def test_an_undeclared_second_contribution_fails_loudly(self, readiness_cls):
        """The negative case: a default expectation of one closes the window early.

        This is the defect the multiplicity exists to prevent. The window -- and
        with it the reshard and reduce-scatter -- closes after the PreProcessNode
        alone, and the MTP node's contribution then arrives as a surplus and is
        raised rather than silently absorbed.
        """
        key = ("embedding.weight",)
        readiness = readiness_cls({key: 1})
        readiness.mark(key)
        assert readiness.is_complete()  # closed early: the reduce would fire here
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(key)

    def test_reset_rearms_the_window_and_keeps_the_declaration(self, readiness_cls):
        """The same parameter owes the same count on every iteration."""
        key = ("embedding.weight",)
        readiness = readiness_cls({key: 2})
        for _ in range(3):
            readiness.mark(key)
            readiness.mark(key)
            assert readiness.is_complete()
            readiness.reset()
            assert not readiness.is_complete()
        assert readiness.expected == {key: 2}

    def test_the_declaration_is_a_live_mapping(self, readiness_cls):
        """``set_grad_multiplicity`` raises the bar in place, before the window opens."""
        embedding, norm = ("embedding.weight",), ("norm.weight",)
        readiness = readiness_cls({embedding: 1, norm: 1})
        assert readiness.expected_total == 2
        readiness.expected.update({embedding: 2})
        assert readiness.expected_total == 3

    def test_an_unknown_parameter_is_a_caller_error(self, readiness_cls):
        """A key outside the declaration must not be swallowed."""
        with pytest.raises(KeyError):
            readiness_cls({("embedding.weight",): 1}).mark(("norm.weight",))


def _build_language_model(tied: bool, mtp_num_layers: int, pipeline_parallel: int = 1):
    """Return a small module carrying the model-level surface the hooks read."""
    from torch import nn

    class LanguageModel(nn.Module):
        """A two-parameter stand-in for a pipeline stage of a language model."""

        def __init__(self) -> None:
            """Build the embedding, the ordinary parameter, and the model flags."""
            super().__init__()
            self.pre_process = True
            self.share_embeddings_and_output_weights = tied
            self.mtp_process = True
            self.config = SimpleNamespace(
                mtp_num_layers=mtp_num_layers, pipeline_model_parallel_size=pipeline_parallel
            )
            self.embedding = nn.Module()
            self.embedding.word_embeddings = nn.Embedding(64, 16)
            self.norm = nn.Linear(16, 16, bias=False)

        def forward(self, tokens):
            """Embed ``tokens`` and project them back to the hidden size."""
            return self.norm(self.embedding.word_embeddings(tokens))

    return LanguageModel()


def _build_tied_language_model(mtp_num_layers: int):
    """Return a language model whose output projection aliases the embedding weight."""
    from torch import nn

    class TiedLanguageModel(nn.Module):
        """A language model with one physical weight under two module paths."""

        def __init__(self) -> None:
            """Tie the output projection to the embedding weight."""
            super().__init__()
            self.pre_process = True
            self.share_embeddings_and_output_weights = True
            self.mtp_process = True
            self.config = SimpleNamespace(
                mtp_num_layers=mtp_num_layers, pipeline_model_parallel_size=1
            )
            self.embedding = nn.Module()
            self.embedding.word_embeddings = nn.Embedding(64, 16)
            self.output_layer = nn.Linear(16, 64, bias=False)
            self.output_layer.weight = self.embedding.word_embeddings.weight

        def forward(self, tokens):
            """Project embedded tokens back to the vocabulary."""
            return self.output_layer(self.embedding.word_embeddings(tokens))

    return TiedLanguageModel()


def _build_mtp_stage_model(
    tied: bool, mtp_num_layers: int, pipeline_parallel: int, pre_process: bool = False
):
    """Return a model shaped like the pipeline stage that owns the MTP loss head.

    That stage owns the output projection and runs MTP, but it is *not* the
    pre-process stage, which is what made the stage flags matter.
    """
    from torch import nn

    class MtpStageModel(nn.Module):
        """An embedding, a distinct output projection, and the MTP flags."""

        def __init__(self) -> None:
            """Build the two weights and the stage flags."""
            super().__init__()
            self.pre_process = pre_process
            self.post_process = True
            self.share_embeddings_and_output_weights = tied
            self.mtp_process = True
            self.config = SimpleNamespace(
                mtp_num_layers=mtp_num_layers, pipeline_model_parallel_size=pipeline_parallel
            )
            self.embedding = nn.Module()
            self.embedding.word_embeddings = nn.Embedding(64, 16)
            self.output_layer = nn.Linear(16, 64, bias=False)
            self.norm = nn.Linear(16, 16, bias=False)
            if tied:
                self.output_layer.weight = self.embedding.word_embeddings.weight

        def forward(self, tokens):
            """Run the embedding through the output projection."""
            return self.output_layer(self.embedding.word_embeddings(tokens))

    return MtpStageModel()


@_needs_distributed
class TestDeclaredMultiplicityOnRealFsdpParameters:
    """What ``register_combined_1f1b_hooks`` declares for a real FSDP unit."""

    @staticmethod
    def _declared(builder, setup):
        """``fully_shard`` a language model, register the hooks, return the model."""
        import torch
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard

        from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
            Placements,
            fully_shard,
            fully_shard_context,
        )
        from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
        from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
            register_combined_1f1b_hooks,
        )

        torch.manual_seed(setup.rank)
        model = builder().to(setup.device)
        mesh = init_device_mesh(setup.device.type, (setup.world_size,))
        placements = Placements(
            dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
        )
        with fully_shard_context(device=setup.device):
            # ``register_hooks=False`` mirrors production, where the combined
            # scheduler installs its own completion hooks instead of the default.
            fully_shard(model, mesh=mesh, placements=placements, register_hooks=False)
        assert isinstance(model, FsdpModule), "fully_shard did not attach the FSDP mixin"
        register_combined_1f1b_hooks(model)
        return model

    @pytest.mark.internal
    @pytest.mark.parametrize(
        ("tied", "mtp_num_layers", "expected_costs"),
        [(True, 1, [1, 3]), (False, 1, [1, 2]), (True, 0, [1, 2]), (False, 0, [1, 1])],
    )
    def test_the_unit_declares_the_embedding_costs(
        self, distributed_setup, tied, mtp_num_layers, expected_costs
    ):
        """Only the embedding accrues extra consumers: +1 per MTP layer, +1 if tied."""
        model = self._declared(
            lambda: _build_language_model(tied, mtp_num_layers), distributed_setup
        )

        declared = model._param_grad_readiness.expected

        assert sorted(declared.values()) == expected_costs, f"declared {declared}"
        embedding_keys = [fqns for fqns in declared if _EMBEDDING_HINT in " ".join(fqns)]
        assert len(embedding_keys) == 1, f"embedding fqns not identified in {declared}"
        assert declared[embedding_keys[0]] == expected_costs[-1]

    @pytest.mark.internal
    def test_a_tied_weight_is_one_parameter_and_costs_three(self, distributed_setup):
        """A tied weight is one ``FsdpParameter`` under two FQNs, so ``len(fqns)`` is not its cost.

        This is the second historical defect: counting the physical parameter once
        per FQN would declare two contributions per consuming node and hold the
        window open forever.
        """
        model = self._declared(lambda: _build_tied_language_model(1), distributed_setup)
        parameters = list(model._trainable_fsdp_parameters())

        assert len(parameters) == 1, f"the tied weight split into {parameters}"
        assert len(parameters[0].fqns) == 2, f"expected two FQNs, got {parameters[0].fqns}"
        # 1 base + 1 MTP pre-dispatch node + 1 tied output projection.
        assert model._param_grad_readiness.expected == {parameters[0].fqns: 3}

    @pytest.mark.internal
    def test_the_declaration_lands_on_real_fsdp_parameters(self, distributed_setup):
        """The unit owns real ``FsdpParameter`` objects, one of them the embedding.

        This pins the assumption the match relies on: the weight reachable through
        the module tree is one of the two objects FSDP swaps between, so an identity
        test against either half recognises it.
        """
        from torch import nn

        from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
            FsdpParameter,
        )

        model = self._declared(lambda: _build_language_model(True, 1), distributed_setup)
        parameters = list(model._trainable_fsdp_parameters())

        assert len(parameters) == 2, f"unexpected parameter count: {parameters}"
        assert all(isinstance(parameter, FsdpParameter) for parameter in parameters)
        assert all(isinstance(parameter.unsharded, nn.Parameter) for parameter in parameters)
        assert all(isinstance(parameter.sharded, nn.Parameter) for parameter in parameters)

        tree_weight = model.embedding.word_embeddings.weight
        assert any(
            parameter.unsharded is tree_weight or parameter.sharded is tree_weight
            for parameter in parameters
        ), "the module-tree weight is neither FSDP object"
        assert set(model._param_grad_readiness.expected) == {
            parameter.fqns for parameter in parameters
        }

    @pytest.mark.internal
    @pytest.mark.parametrize(
        ("pipeline_parallel", "mtp_num_layers", "expected_output_cost"),
        [(1, 1, 1), (2, 1, 2), (2, 0, 1)],
    )
    def test_the_output_projection_costs_two_on_the_interleaved_schedule(
        self, distributed_setup, pipeline_parallel, mtp_num_layers, expected_output_cost
    ):
        """The MTP loss head is its own schedule node only when the schedule is interleaved.

        ``model_chunk_schedule_plan`` builds ``mtp_post_process`` -- which contains the
        output projection -- for every MTP layer. On the interleaved schedule that
        node's backward is its own GraphTask, so the projection's weight is consumed
        twice per window; on the PP=1 no-pipelining path it shares one GraphTask with
        the main projection and is consumed once. Declaring the extra consumer at
        PP=1 would hold the window open forever.
        """
        model = self._declared(
            lambda: _build_mtp_stage_model(
                False, mtp_num_layers, pipeline_parallel, pre_process=True
            ),
            distributed_setup,
        )

        declared = model._param_grad_readiness.expected
        output_keys = [fqns for fqns in declared if _OUTPUT_HINT in " ".join(fqns)]

        assert len(output_keys) == 1, f"output layer fqns not identified in {declared}"
        assert declared[output_keys[0]] == expected_output_cost

    @pytest.mark.internal
    @pytest.mark.parametrize(("pipeline_parallel", "expected_cost"), [(1, 3), (2, 4)])
    def test_a_tied_weight_on_an_mtp_stage_pays_for_the_projection(
        self, distributed_setup, pipeline_parallel, expected_cost
    ):
        """``pre_process`` alone is the wrong gate for ``tied``.

        The MTP stage owns the output projection and drives it through
        ``mtp_process`` while ``pre_process`` is False. Gating ``tied`` on
        ``pre_process`` dropped the projection's contribution, and the embedding
        over-fired ``3 > 2`` on a PP2/VPP2 run. On the interleaved schedule the
        projection is contributed to a second time, and because it runs against this
        very weight object the embedding pays for that as well.
        """
        model = self._declared(
            lambda: _build_mtp_stage_model(True, 1, pipeline_parallel), distributed_setup
        )

        declared = model._param_grad_readiness.expected
        embedding_keys = [fqns for fqns in declared if _EMBEDDING_HINT in " ".join(fqns)]

        assert len(embedding_keys) == 1, f"embedding fqns not identified in {declared}"
        # 1 base + 1 MTP pre-dispatch node + 1 tied output projection, plus the
        # projection's second interleaved contribution when the schedule is split.
        assert declared[embedding_keys[0]] == expected_cost
