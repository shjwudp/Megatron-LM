# Proxy memory calculator

The calculator exposes exactly two independent proxy presets: `hfsdp16` (default) and
`local` (EP2). The presets are calculation examples, not production model descriptions.
There is no parent production preset, hidden CLI preset or production-sized golden test.

Download or copy `memory_calculator.html` and open it directly. Presets, buffer
accounting and estimation logic are embedded, so calculation and configuration
import/export work offline without sibling JavaScript files. The linked Markdown
guide is optional documentation, not a calculation dependency.

The standalone JavaScript files remain the source for both the browser and Node.
After editing either model, refresh the embedded copies and check their consistency:

```bash
node tools/proxy_memory_calculator/bundle_calculator.js
node tools/proxy_memory_calculator/bundle_calculator.js --check
```

For command-line estimates and model regression checks, run:

```bash
node tools/proxy_memory_calculator/memory_model.js hfsdp16
node tools/proxy_memory_calculator/memory_model.js local
node --test tools/proxy_memory_calculator/test_memory_model.js
```

## Supported path and topology

This model covers the CSA proxy path with BF16/MXFP8, packed THD, TP1, CP, VPP and
MFSDPv2. Routed experts use ZeRO-1; the dense selector compares ZeRO-1/2/3. Unsupported
configurations fail validation rather than silently using a different model.
World is derived from EDP multiplier × PP × EP. Legacy World-only JSON infers the
multiplier. Decoder layer count comes from the expanded pipeline layout; MTP is counted
separately. Each preset keeps its own explicit GBS/group. The automatic-batch button uses
G=PP and four microbatch groups; manual values remain explicit overrides.

## Static and dynamic memory

Static memory includes local compute weights, FP32 main weights, optimizer states and
main gradients with dense/expert mesh placements. MXFP8 row/column planes and scaling
storage are counted. Optional padded shards use source row-atomic/LCM alignment; they
can substantially increase compute/master/gradient buffers. ZeRO-1 comparisons represent
the unpadded baseline. Padding comparisons use the same PP rank on both sides.

Saved activations follow microbatch/chunk forward-backward order. CSA Q/KV, compressor,
indexer, grouped output, dense/expert MLP, loss and MTP are modeled separately. Aliased
FlashMLA O/grouped-projection input is counted once. The UI marks original versus added
recompute options. Output replay removes saved O/LSE across outstanding work but still
requires a single-layer backward workspace. Fused indexer gradients produced in forward
are not removed by dsa_selected_scores.

Allocated estimates add bounded temporary work: compute workspace, parameter gather,
two full FSDP unit staging buffers and one unit of autograd wgrad. Expert-DP size one
still packs gradients. Root dtype buckets and expert units are counted before sharding.
These maxima are an envelope, not an exact simultaneous allocation schedule; final-stage
predictions can be conservative. Full Muon Newton–Schulz workspace is not modeled.

Budget = estimated allocated + fragmentation allowance + non-Torch allowance.
The fragmentation field is an assumption, not measured Torch reserved memory. Real
NCCL/context use, allocator fragmentation, token imbalance and input lengths can change
peaks. Compare max allocated with allocated estimates and sampled device peaks with
budget separately. A device sample is not an exact instantaneous global maximum.

## Calibration and verification

The compact JSON fixtures retain only proxy configuration and numeric values needed by
tests. Raw job identities, original paths and production comparisons remain outside Git.
The 16-node proxy check compares estimates with measured per-PP allocator maxima within
5% and verifies staging against live allocations at a representative snapshot peak.
This is historical calibration, not a new 16-node experiment. EP2 fixtures predate the
compute-stream staging change, so those checks compare the corresponding base estimate.
Tests additionally cover replay, aliases, EDP, padding, ZeRO/gather, manual batches and
preset isolation. Browser checks cover both presets, errors, import/export and comparisons.

Training launchers and cluster validation configuration are intentionally separate from
this calculator. It never edits a training configuration or submits a job.
