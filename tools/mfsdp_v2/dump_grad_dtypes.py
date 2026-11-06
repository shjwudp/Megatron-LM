"""Debug entry for pretrain_gpt.py: print gradient dtypes along the MFSDP v2 reduction path once per class.

Shows the autograd grad produced by backward, the reduce-scatter input (partial_grad),
the persistent accumulator (main_grad) and the optimizer-visible sharded grad.
"""
import os
import runpy
import sys

import megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group as pgm

def _report_ranks():
    """Ranks that print, from MFSDP_DEBUG_RANKS (comma-separated, default "0")."""
    return {int(r) for r in os.environ.get("MFSDP_DEBUG_RANKS", "0").split(",") if r}


_seen = set()
_orig_copy = pgm.FsdpParameterGroup.copy_gradients_to_partial_buffer


def _copy(self, partial_grad):
    if int(os.environ.get("RANK", "0")) in _report_ranks():
        cls = type(self._owning_module()).__name__
        kind = "expert" if "GroupedMLP" in cls or "SequentialMLP" in cls else "dense"
        if kind not in _seen:
            _seen.add(kind)
            g = self.fsdp_parameters[0].unsharded.grad
            print(
                f"[graddtype] r{os.environ.get('RANK')} {kind:6s} module={cls} "
                f"autograd_grad={g.dtype} partial_grad(RS in)={partial_grad.dtype} "
                f"main_grad={self.main_grad.dtype} placements={self.main_grad.placements}",
                flush=True,
            )
    return _orig_copy(self, partial_grad)


pgm.FsdpParameterGroup.copy_gradients_to_partial_buffer = _copy
sys.argv[0] = "pretrain_gpt.py"
runpy.run_path("pretrain_gpt.py", run_name="__main__")
