# Plan

1. Audit the reference source, optimizer defaults and current run configurations.
2. Fix MFSDP configuration forwarding and implement shared logical splits.
3. Check configured optimizer construction, independent NS mathematics and
   distributed loss/weight equivalence.
4. Train both paths locally on eight GPUs with packed input, VPP, EP overlap,
   deferred weight gradients and selective recomputation. Audit actual optimizer
   objects and parameter ownership on every rank.
5. Freeze the reviewed code and observation harness. Run both larger comparisons from the same immutable source and runtime bundle.
6. Regenerate throughput, memory and Event Tracker/nsys analysis after completion.
