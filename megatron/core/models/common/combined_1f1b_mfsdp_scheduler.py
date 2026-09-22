# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    COLWISE,
    ROWWISE,
)
from megatron.core.utils import get_attr_wrapped_model


def _make_unshard_forward_hook(owner: FsdpModule):
    """Forward pre-hook: unshard the owning FSDP module before the submodule forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        # A forward GEMM consumes the row-wise MXFP8 payload. If the same
        # materialization also has to serve a backward pass (activation
        # recomputation with no reshard between them), the module widens it.
        owner.unshard(orientation=ROWWISE)

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Backward pre-hook: unshard the owning FSDP module before the submodule backward."""

    def hook(submodule, _grad_output):
        # The backward GEMM consumes the column-wise MXFP8 payload.
        owner.unshard(orientation=COLWISE)

    return hook


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."

    # The FSDP unit is an ``ExperimentalFsdpFloat16Module``: MCore applies the
    # mixed-precision wrapper before FSDP, and ``Float16Module`` defines no
    # ``__getattr__``, so the real GPTModel stays at ``.module``. The model-level
    # attributes the multiplicity contract needs -- ``pre_process``,
    # ``share_embeddings_and_output_weights``, ``mtp_process`` and ``embedding`` --
    # live on that inner model, so resolve it through the shared unwrapping helper
    # instead of reading them off the wrapper.
    # This runs once per model chunk, so the declaration below is derived from that
    # chunk's own ``pre_process``/``post_process``/``mtp_process``.
    model = get_attr_wrapped_model(module, "pre_process", return_model_obj=True)
    register_hooks(module, module)

    # Tied embeddings: the output projection borrows the embedding weight and owns no
    # parameter of its own -- ``gpt_model`` passes
    # ``skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights``
    # and builds the projection with ``bias=False``, so ``named_parameters`` never reaches
    # a weight for it. ``register_hooks`` above therefore skips it, and nothing
    # re-materializes the borrowed weight before the PostProcessNode's backward, the first
    # backward consumer. That backward then runs against storage the post-forward reshard
    # already released:
    #   RuntimeError: The tensor has a non-zero number of elements, but its data is not
    #   allocated yet.   (tensor_parallel/layers.py:748)
    # ``dbuffer.release_storage`` is not at fault -- it keeps the Storage object precisely
    # so a later reallocate restores saved aliases; the missing piece is that re-unshard.
    # Register the two hooks on that single module, pointed at the unit that owns the
    # borrowed weight (unshard/reshard act on an FSDP unit, not on a module). This is
    # deliberately narrow: it runs only when the weights are tied *and* the projection
    # really owns nothing, so the untied path registers exactly what it registered before
    # and no module gains a redundant unshard.
    if model.share_embeddings_and_output_weights:
        output_layer = getattr(model, 'output_layer', None)
        if output_layer is not None and not list(output_layer.parameters(recurse=False)):
            embedding = getattr(model, 'embedding', None)
            weight_owner_module = getattr(embedding, 'word_embeddings', None) or embedding
            borrowed_owner = _nearest_fsdp_owner(module, weight_owner_module)
            if borrowed_owner is not None:
                output_layer.register_forward_pre_hook(
                    _make_unshard_forward_hook(borrowed_owner), prepend=True, with_kwargs=True
                )
                output_layer.register_full_backward_pre_hook(
                    _make_unshard_backward_hook(borrowed_owner)
                )
    mtp_depth = _active_mtp_layers(model)
    # The output projection runs against the shared embedding weight whenever this
    # chunk feeds that weight to the projection, so the projection's contribution
    # has to be declared on the embedding. ``pre_process`` alone is the wrong gate:
    # on an MTP stage the projection is driven by ``mtp_process`` while
    # ``pre_process`` is False, and gating on ``pre_process`` left the embedding one
    # contribution short -- a PP2/VPP2 MTP run over-fired 3 > 2 on the embedding.
    tied = model.share_embeddings_and_output_weights and (model.pre_process or model.mtp_process)
    # A chunk that runs MTP schedules its loss head as its own node:
    # ``model_chunk_schedule_plan`` builds ``mtp_post_process`` for every MTP layer,
    # and that node contains the output projection. On a schedule that runs each
    # node's backward as its own GraphTask the output projection's weight is
    # therefore consumed twice per window. ``schedules.py:161-167`` selects the
    # interleaved schedule only when ``pipeline_model_parallel_size > 1``; the PP=1
    # no-pipelining path keeps both projections inside one GraphTask, so the extra
    # consumer must not be declared there.
    interleaved = model.config.pipeline_model_parallel_size > 1
    embedding_weight = getattr(
        getattr(getattr(model, 'embedding', None), 'word_embeddings', None), 'weight', None
    )
    output_weight = getattr(getattr(model, 'output_layer', None), 'weight', None)
    # The weights below are contributed to twice per window on the interleaved
    # schedule: the nodes that consume them run as their own GraphTasks there, while
    # the PP=1 no-pipelining schedule keeps them inside an existing GraphTask, so the
    # extra consumer must not be declared at PP=1. They are the modules on the path
    # from the last decoder layer to the loss -- the decoder's final layernorm, the
    # output projection, and the MTP layer's own weights -- because that path is
    # scheduled as ``post_process``/``mtp_post_process`` nodes, each of which consumes
    # its module once per window. The decoder's final layernorm is easy to miss: it is
    # an ordinary decoder parameter, but with MTP it feeds both the main loss path and
    # the MTP head, so it is consumed twice like the projection is. Declaring it once
    # was measured to be one short -- PP2/VPP2 with MTP over-fired on it ``2 > 1``
    # (job 19052914) and completes once it is declared twice (job 19055665). When the
    # output projection runs against the embedding weight, that weight is in this set
    # as well and picks the extra consumer up on top of the embedding's own terms;
    # ``_unit_grad_multiplicity`` applies both.
    mtp_post_weights = ()
    if mtp_depth and interleaved:
        mtp_post_weights = tuple(
            weight
            for weight in (
                output_weight,
                *_mtp_layer_weights(model),
                getattr(
                    getattr(getattr(model, 'decoder', None), 'final_layernorm', None),
                    'weight',
                    None,
                ),
            )
            if weight is not None
        )
    for submodule in module.modules():
        if not isinstance(submodule, FsdpModule):
            continue
        submodule.set_grad_multiplicity(
            multiplicity=_unit_grad_multiplicity(
                submodule, mtp_depth, embedding_weight, mtp_post_weights, tied
            )
        )
        submodule.register_post_backward_hook(_module_post_backward_hook)


def _nearest_fsdp_owner(root, target):
    """Return the ``FsdpModule`` whose parameter tree contains ``target``.

    The unshard hooks act on an FSDP unit, not on a module, so a module that borrows
    another module's parameter -- the tied output projection borrowing the embedding
    weight -- has to be pointed at the unit that owns that parameter rather than at its
    own nearest unit. Returns ``None`` when ``target`` is not in ``root``'s tree, which is
    the case on a chunk that does not carry the embedding at all.
    """
    if target is None:
        return None
    found = []

    def walk(node, owner):
        if isinstance(node, FsdpModule):
            owner = node  # BEFORE descending: an FSDP unit owns itself
        if node is target:
            found.append(owner)
            return
        for child in node.children():
            walk(child, owner)

    walk(root, None)
    return found[0] if found else None


def _active_mtp_layers(module) -> int:
    """Return whether THIS pipeline stage runs MTP (0 or 1)."""
    depth = getattr(module.config, 'mtp_num_layers', None) or 0
    if depth == 0:
        return 0
    if not hasattr(module, 'mtp_process'):
        raise AssertionError(
            "config.mtp_num_layers is set but the model exposes no `mtp_process`; "
            "cannot tell whether this pipeline stage runs MTP, and guessing would "
            "produce a wrong gradient multiplicity."
        )
    if not module.mtp_process:
        return 0
    assert depth == 1, (
        "overlap_moe_expert_parallel_comm requires mtp_num_layers <= 1 "
        "(transformer_config.py:3316-3320); per-parameter multiplicity does not "
        f"model deeper MTP (got {depth})."
    )
    return 1


def _matches_fsdp_parameter(fsdp_parameter, weight) -> bool:
    """Return whether ``fsdp_parameter`` is ``weight``.

    The match is by object identity against *both* objects FSDP swaps between.
    ``parameter_group._set_module_parameter`` is the only writer of
    ``module._parameters``, and it installs either ``FsdpParameter.sharded`` or
    ``FsdpParameter.unsharded``, so the weight read through the module tree is one
    of those two -- which one depends on whether the last switch was a reshard or an
    unshard. Matching only ``unsharded`` therefore never fires in practice: an MTP
    run declared the embedding once while its hook fired twice (PreProcessNode plus
    the MTP pre-dispatch node) and the over-fire guard raised.

    ``None`` must match nothing. It is ruled out explicitly rather than left to the
    identity tests, because a parameter whose ``sharded``/``unsharded`` is unset
    would otherwise compare equal to it and hand every parameter the extra
    consumers.
    """
    if weight is None:
        return False
    return fsdp_parameter.unsharded is weight or fsdp_parameter.sharded is weight


def _mtp_layer_weights(model):
    """Return the weights ``MultiTokenPredictionLayer`` owns directly.

    These are the MTP-specific weights: the ``enorm``/``hnorm``/``eh_proj`` that
    ``_concat_embeddings`` uses and the ``final_layernorm`` that ``_postprocess``
    uses. On the interleaved schedule each of them is contributed to twice per
    window, while the parameters of the ordinary transformer layers -- including
    the layer ``mtp_model_layer`` borrows -- are contributed to once, so that
    borrowed subtree is excluded here.
    """
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    weights = []
    for submodule in model.modules():
        if not isinstance(submodule, MultiTokenPredictionLayer):
            continue
        for name, child in submodule.named_children():
            if name == 'mtp_model_layer':
                continue
            weights.extend(child.parameters())
    return tuple(weights)


def _unit_grad_multiplicity(
    unit, mtp_depth: int, embedding_weight, mtp_post_weights, tied: bool
) -> dict:
    """Per-parameter backward contribution counts for the combined 1F1B path.

    Nearly every parameter's gradient comes from exactly one schedule node, so the
    default is 1. Two groups have extra consumers:

    The embedding:

      * this chunk's PreProcessNode embedding lookup          -> the base 1
      * one MTP pre-dispatch node per MTP layer              -> +mtp_depth
      * the output projection, when it runs against this very weight object
        (``tied``)                                           -> +1

    The MTP nodes' own weights -- the output projection and each MTP layer's
    ``enorm``/``hnorm``/``eh_proj``/``final_layernorm``. Those nodes run as their own
    GraphTasks on the interleaved schedule, so each weight is contributed to twice
    per window. ``mtp_post_weights`` carries exactly that set, and is empty when the
    extra consumer does not exist (PP=1, or no MTP on this stage). A weight can be in
    that set *and* be the embedding, which is why the two contributions are added
    independently rather than as alternatives.

    The extra consumers are decided by the caller from this model chunk's stage
    flags and the pipeline size, because they are not derivable from the FSDP unit
    alone.
    """
    mtp_extra = 1 if mtp_post_weights else 0
    multiplicity = {}
    for fsdp_parameter in unit._trainable_fsdp_parameters():
        consumers = 1
        if _matches_fsdp_parameter(fsdp_parameter, embedding_weight):
            consumers += mtp_depth + (1 if tied else 0)
            if tied:
                # When the output projection runs against this weight, the embedding
                # inherits the projection's extra interleaved contribution as well.
                # Object identity cannot decide that here: on an MTP stage the output
                # layer allocates its own parameter, so only the shared-weight flag
                # tells us the projection runs against the embedding weight.
                consumers += mtp_extra
        elif any(_matches_fsdp_parameter(fsdp_parameter, weight) for weight in mtp_post_weights):
            consumers += mtp_extra
        multiplicity[fsdp_parameter.fqns] = consumers
    return multiplicity
