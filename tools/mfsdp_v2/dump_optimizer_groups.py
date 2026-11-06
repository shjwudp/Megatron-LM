"""Debug entry for pretrain_gpt.py: print which optimizer owns each parameter, then train.

Works for DistOpt (maps shard-local main params back through model_param_group_index_map),
MFSDP v2 (FullyShardedOptimizer._param_to_fqn) and ChainedOptimizer. Prints on MFSDP_DEBUG_RANKS.
"""
import os
import runpy
import sys

import megatron.core.optimizer as mo

def _report_ranks():
    """Ranks that print, from MFSDP_DEBUG_RANKS (comma-separated, default "0")."""
    return {int(r) for r in os.environ.get("MFSDP_DEBUG_RANKS", "0").split(",") if r}


_orig = mo.get_megatron_optimizer


def _wrapped(config, model_chunks, *args, **kwargs):
    optimizer = _orig(config, model_chunks, *args, **kwargs)
    if int(os.environ.get("RANK", "0")) in _report_ranks():
        name_of = {}
        for chunk in model_chunks:
            for name, param in chunk.named_parameters():
                name_of[param] = name
                # Megatron-FSDP v1 / DistOpt may hand the optimizer a main-weight copy.
                for attr in ("main_param", "_mfsdp_main_param"):
                    if getattr(param, attr, None) is not None:
                        name_of[getattr(param, attr)] = name
        opts = getattr(optimizer, "chained_optimizers", [optimizer])
        for opt in opts:
            inner = opt.optimizer
            kind = type(inner).__name__
            fqns = getattr(opt, "_param_to_fqn", None) or {}
            # DistributedOptimizer: shard-local main params; map back through its model params.
            index_map = getattr(opt, "model_param_group_index_map", None)
            if index_map is not None:
                for model_param, (group_index, _) in index_map.items():
                    print(
                        f"[optmap] {kind:12s} {name_of.get(model_param, '?')} "
                        f"{tuple(model_param.shape)} group={group_index} "
                        f"opt={inner.param_groups[group_index].get('optimizer', '-')}",
                        flush=True,
                    )
                continue
            for group in inner.param_groups:
                for p in group["params"]:
                    name = name_of.get(p) or fqns.get(p) or getattr(p, "param_name", "?")
                    print(f"[optmap] r{os.environ.get('RANK')} {kind:12s} {name} {tuple(p.shape)} local={tuple(p.to_local().shape) if hasattr(p, 'to_local') else None}", flush=True)
    return optimizer


mo.get_megatron_optimizer = _wrapped
import megatron.training.training as tr  # noqa: E402

tr.get_megatron_optimizer = _wrapped
sys.argv[0] = "pretrain_gpt.py"
runpy.run_path("pretrain_gpt.py", run_name="__main__")
