# Separate FSDP adapters for VPP and combined 1F1B

This is a CPU-runnable design prototype, outside `megatron/core`. It introduces
no imports from Megatron or PyTorch, no hooks, and no changes to `ExecutionOrder`,
`FsdpContext`, `FsdpModule`, or existing pipeline/combined schedulers.

The goal is to review **who decides parameter communication** before integrating
anything into the runtime. There is no execution recording, replay, global trace,
or schedule inference.

## Conventional VPP: select a pass, then prefetch inside that pass

`VppFsdpScheduler` consumes the existing pipeline scheduler's
`Step(chunk, microbatch, phase)` decisions. It does not generate the VPP schedule,
implement warmup/cooldown, execute compute, or perform P2P communication.

Each `Chunk` supplies explicit forward and backward unit occurrences. Backward
order is supplied independently because nested modules need not be the reverse
of forward order. Repeated units are allowed in either tuple.

```python
with adapter.iteration(forward_only=False):
    # The pipeline scheduler already selected chunk=0, microbatch=1.
    with adapter.run_pass(Step(0, 1, 'forward')):
        adapter.demand(root0)   # materialize root0; prefetch layer0
        # root compute / existing module calls
        adapter.demand(layer0)
        # layer compute / existing module calls
```

A fresh cursor belongs to this one pass. Executing chunk 0 for another microbatch
starts another pass at position zero; switching chunks selects another order.
There is no shared construction-order cursor to resynchronize and no
`_unique_index`. Prefetch stops at the end of the pass; cross-pass prefetch is
intentionally deferred. Root hooks do not decide pass boundaries.

Existing runtime callbacks retain release and gradient-readiness ownership. The
adapter validates forward/backward pass pairing and demand order, not gradient
readiness or optimizer safety. Delayed gradients still require the runtime's
normal completion/synchronization before an optimizer step. Forward-only
validation uses its own iteration scope and does not consume backward order.

## Combined 1F1B: explicit actions on compute nodes

`Combined1f1bFsdpScheduler` consumes nodes chosen by the existing fine-grained
compute plan. Each `Node` explicitly lists:

- units to demand before compute;
- units to prefetch while compute runs;
- units to release after successful compute;
- units whose backward can be finalized after successful compute.

```python
with adapter.iteration():
    with adapter.run_node(Node(Step(1, 0, 'backward'), 'B.mlp', demand=(unit_b,))):
        backward_mlp()
    with adapter.run_node(Node(Step(0, 1, 'forward'), 'A.attention', demand=(unit_a,))):
        forward_attention()
    with adapter.run_node(Node(Step(1, 0, 'weight_grad'), 'B.wgrad', finish_backward=(unit_b,))):
        compute_delayed_weight_gradients()
```

Returning from `B.mlp` does not infer that gradients are ready. Its terminal wgrad
node explicitly invokes backward finalization. The runtime callback must enforce
actual readiness and owns reduction plus final parameter release. A unit cannot
also appear in that node's release list.

The sketch conservatively allows one unfinished backward window per unit. Reusing
the same unit in another microbatch's backward before finalizing the earlier
window is rejected. Supporting overlapping gradient windows would require an
explicit runtime/buffer contract, rather than merely counting node executions.

## Relationship to the current combined FSDP hooks

The upstream `combined_1f1b_mfsdp_scheduler.py` already separates custom FSDP
communication from the compute plan. The prototype follows that ownership split.
A future integration can bind these adapters' injected `FsdpActions` to that
runtime, rather than inserting a general-purpose trace into core FSDP.

`demand` must materialize and wait for the unit; `prefetch` must issue a nonblocking,
idempotent gather. `release` must respect CUDA consumers. `finish_backward` must
honor gradient readiness and communication ordering. These are callback contracts,
not implementations supplied by this prototype. They must not be layered on top
of another hook path that independently performs the same communication.

## Run and limitations

From the repository root, with ordinary Python and no installed dependencies:

```bash
python3 -m examples.megatron_fsdp.scheduler_prototype.demo
python3 -m unittest examples.megatron_fsdp.scheduler_prototype.test_schedulers -v
```

The demo prints callback order. Three tests check grouped VPP revisits and
validation, interleaved combined nodes with delayed wgrad, and invalid windows.
They use a fake materialization/gradient-ready runtime; they do not validate GPU
collectives, numerical training, P2P, stream ordering, memory, or overlap.
Failures can leave runtime work incomplete; discard failed work rather than
assuming adapter bookkeeping recovers model state.

The proposed review boundary is these separate scheduling policies and their
runtime callback contract. Actual scheduler wiring, hook replacement, CUDA graph
support, and performance validation are later integration work.
