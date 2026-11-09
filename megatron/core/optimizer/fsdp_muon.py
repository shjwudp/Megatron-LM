# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MCore configuration and logical matrix splits for the MFSDP Muon optimizer."""

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.muon import FsdpMuon

from .emerging_optimizers import _get_qkv_split_shapes, validate_coefficient_type
from .muon_matrix import MuonMatrixOrthogonalizer


class MCoreFsdpMuon(FsdpMuon):
    """Keep FSDP's owner scheduling, with the same Muon math as TP=1 DDP.

    Map config explicitly: inspecting FsdpMuon's (*args, **kwargs) signature
    silently drops every optimizer setting.
    """

    def __init__(self, params, config, model_config):
        validate_coefficient_type(config.muon_coefficient_type)
        self.matrix_orthogonalizer = MuonMatrixOrthogonalizer(
            coefficient_type=config.muon_coefficient_type,
            num_ns_steps=config.muon_num_ns_steps,
            scale_mode=config.muon_scale_mode,
            extra_scale_factor=config.muon_extra_scale_factor,
            ns_eps=config.muon_ns_eps,
            split_qkv=config.muon_split_qkv,
            split_head=config.muon_split_head,
            split_fc1=config.muon_split_fc1,
            split_eh=config.muon_split_eh,
            qkv_split_shapes=tuple(_get_qkv_split_shapes(model_config)),
            use_syrk=config.muon_use_syrk,
        )
        super().__init__(
            params,
            lr=config.lr,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            weight_decay=config.weight_decay,
            weight_decay_method="decoupled",
            fp32_matmul_prec=config.muon_fp32_matmul_prec,
            coefficient_type=config.muon_coefficient_type,
            num_ns_steps=config.muon_num_ns_steps,
            scale_mode=config.muon_scale_mode,
            extra_scale_factor=config.muon_extra_scale_factor,
            use_syrk=config.muon_use_syrk,
        )

    def orthogonalize(self, p, grad, **kwargs):
        """Run on the assembled full matrix at its FSDP owner."""
        return self.matrix_orthogonalizer(p, grad)
