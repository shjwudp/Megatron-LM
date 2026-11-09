# MFSDPv2 Muon and Adam for a proxy model

The optimizer registry routes trainable non-embedding matrices to Muon, including
router weights, attention projections, dense/shared/routed expert projections and
MTP EH. Embeddings, output weights, one-dimensional parameters and explicitly
excluded parameters use Adam or the selected scalar optimizer. CSA compressor
position-bias tables remain excluded even though their storage is two-dimensional.
The MTP embedding replica also follows the embedding rule.

## Effective configuration

`MCoreFsdpMuon` forwards configuration explicitly. Inferring arguments from the
underlying `FsdpMuon(*args, **kwargs)` signature previously returned no settings:
NS steps, precision, Nesterov and update scale silently fell back to EO defaults.
Learning rate and weight decay could still be supplied by parameter groups and
the scheduler; those were not evidence of successful constructor forwarding.

The additional reference algorithm is selected explicitly:

```text
--muon-coefficient-type r2s10 --muon-num-ns-steps 10
--muon-momentum 0.95 --muon-nesterov
--muon-fp32-matmul-prec high --muon-extra-scale-factor 0.2
--muon-scale-mode spectral --muon-ns-eps 1e-15
--muon-no-split-qkv --muon-split-head --muon-split-fc1 --muon-split-eh
--muon-tp-mode duplicated
```

For Adam, the comparison uses beta1=0.9, beta2=0.95 and epsilon=1e-20.
The new options do not change repository defaults for existing DDP runs.

## Logical matrix splits

Both implementations use `MuonMatrixOrthogonalizer` on a complete matrix.
MFSDP assembles it on the existing owner before orthogonalization. DDP uses the
owner's complete FP32 master. Splitting adds no communication.

| Parameter | Orthogonalization |
| --- | --- |
| Fused QKV | Per head when head splitting is enabled; QKV component splitting takes precedence if also enabled |
| CSA indexer Q (`linear_wq_b`) | Per indexer head, along output rows |
| Dense/shared/routed FC1, including grouped `weightN` | Gate/up halves along output rows |
| MTP EH | Embedding/hidden halves along input columns |
| Main MLA Q low-rank projections, grouped O projections and compressor matrices | Whole matrix |

The scale is computed from each block's dimensions. The r2s10 coefficient schedule
repeats its final coefficient for requests exceeding ten steps. An explicit NS
epsilon applies to the normalization of every block. Metadata survives DDP master
creation; MFSDP tags the sharded Parameters after wrapping and reads them at the
owner update. The precision context is still managed by the optimizer step.

CSA contains parameters absent from the reference attention implementation. The
indexer Q mapping is explicit; other new matrices use the whole-matrix policy.
The existing weight-decay rule is retained: the Q-layernorm name matches the
Q/K-layernorm exemption override, while the newly named KV latent norm does not.
APE's Adam exclusion does not itself exempt it from weight decay.

## Sharding and scope

Owner gather/scatter supports replicated/sharded hybrid meshes and fully sharded
N-D meshes. Work assignment remains grouped by mesh so dense and expert groups
may differ. Momentum remains sharded; complete update matrices exist only on
assigned owners. All participating ranks, including empty local shards, retain
the established communication sequence.

The new split/r2s10/epsilon path supports TP=ETP=1 without GTP rematerialization.
Layer-sharded Muon rejects these options. MFSDP v1, vision-specific optimizer
routing and checkpoint conversion are outside this change. The whole-matrix
policy is a proxy model migration rule, not a claim of identical architectures.

## Validation

Validation details are recorded in [the task result](../tasks/proxy-optimizer-alignment/Result.md).
Tests exercise the actual MCore factory, explicit polynomial mathematics, metadata
copying, five-step loss/weight parity under ordinary and hybrid sharding, and the
existing FSDP Muon regressions. RowAtomic final weights are reconstructed through
their DBuffer layout; ordinary `DTensor.full_tensor()` is not a valid reconstruction
for arbitrarily packed tensor ownership.
