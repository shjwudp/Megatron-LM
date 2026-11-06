# Deterministic MoE load forcing for proxy benchmarks

moe_router_force_load_balancing_mode selects the existing random-logit forcing or
round_robin. The default remains random, and no forcing occurs unless the existing
force-load-balancing switch is enabled. This feature changes routing/logits and must
not be described as passive profiling or ordinary training semantics.

Round-robin selects consecutive expert windows with a rank-dependent starting offset.
It has exact counts when token count divides the expert count; uneven/padded token
counts can retain boundary imbalance. The backward preserves the existing straight-
through gradient contract. Group-limited routing and forced bias are rejected.

## Validation

The regression checks equal counts for divisible batches, bounded imbalance otherwise,
and exact identity gradients. Run test_round_robin_balance.py through torchrun in the
supported CUDA environment. New FSDP/DDP proxy benchmarks use the same forcing mode
so expert-load variation does not obscure the comparison.

Reconstructed branch result: both count/straight-through-gradient cases passed.
