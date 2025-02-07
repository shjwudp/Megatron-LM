## How to use ?

Add these flag to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Configuration Recommendataions

CPU Optimizer step, GPU gradient D2H and CPU optimizer updated parameter H2D are quite time-consuming operations, it is recommended to turn on the overlap switch to make them execute in parallel to reduce the end-to-end time. Enable this feature with the flag `--overlap-cpu-optimizer-d2h-h2d`.
