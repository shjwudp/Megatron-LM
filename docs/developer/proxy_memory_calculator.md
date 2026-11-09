# Proxy memory estimation

The estimator, buffer model, UI, calibration corrections and their documentation are
consolidated into one feature. The canonical [calculator guide](../../tools/proxy_memory_calculator/memory_calculator.md)
contains the supported path, formulas, temporary-memory bounds, preset rules, calibration
scope and validation commands. Only the 16-node and EP2 proxy presets are included.

The consolidation combines the FSDP staging/wgrad and EP-overlap corrections with the
editable EDP/batch controls and ZeRO/padding comparisons. Padding deltas include temporary
gradient storage as well as resident buffers and gather allocations. Preset batch values
are preserved until the user requests automatic batch calculation.

Verification: 19 Node regressions passed on the merged implementation. Historical
snapshot values are identified as calibration; reconstructed-branch training and browser
results are recorded in the [branch validation summary](proxy_validation.md).

The downloadable HTML now embeds both JavaScript models. A single-file browser
check with networking disabled verified both presets, editable budgets, ZeRO and
recompute controls, and JSON import/export/download. Its preset estimates matched
the Node model, and all 19 model regressions still passed. After editing a model,
run `node tools/proxy_memory_calculator/bundle_calculator.js`; use `--check` to
verify that the embedded copies match the source files.
