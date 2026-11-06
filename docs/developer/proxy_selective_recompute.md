# Local selective recomputation for a proxy model

The supported CSA path combines existing main-branch selective recompute with four
additional modules: dense_fc1_act, moe_fc1, csa_o_proj and csa_core_output.

| Region | Stored state released | Replay boundary |
| --- | --- | --- |
| Dense FC1 and activation | expanded MLP intermediate | local FC1/activation before FC2 backward |
| Expert FC1 | expert hidden expansion | local grouped GEMM; requires moe_act |
| CSA grouped O projection | projection input/intermediate | local output projection |
| CSA core O/LSE | FlashMLA output and log-sum-exp | local attention replay on already prepared KV |

Attention QKV/norm/RoPE replay uses the existing mla_up_proj option. Shared expert
recomputation excludes nested dense_fc1_act to preserve checkpoint ownership. FP8 FC2
retains original input as required by TE. Checkpoint boundaries order upstream layernorm
restoration before FC1 replay, and expert scale/probability gradients remain connected.

CSA O replay uses the output-discarding helper. CP exchange and compressed-KV preparation
remain outside the replay boundary; this does not repeat CP collectives. The backward
still needs a transient single-layer output. dsa_selected_scores does not reduce saved
state on the fused indexer path, where indexer gradients are prepared in forward.

## Validation

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/test_csa_local_recompute.py --experimental
python -m torch.distributed.run --standalone --nproc-per-node 2 -m pytest -q \
  tests/unit_tests/determinism/kernels/test_csa_output_recompute.py --experimental
```

Tests compare baseline and recomputed outputs, input/weight/router gradients, actual
replay counts, FP8 state and upstream norm ordering. The determinism test checks the CSA
output replay kernel. Existing packed-MTP regressions cover integration boundaries.
Calculator formulas and snapshot tools are maintained in their own consolidated features.

Reconstructed branch: 12 local tests passed; CP2 output-recompute tests passed (4 per rank).
The first CP2 invocation used one process and failed the required topology check; the
correct two-process rerun passed without changing implementation or assertions.
