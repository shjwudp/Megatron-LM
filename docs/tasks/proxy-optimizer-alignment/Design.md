# Design

MFSDP's variadic constructor cannot be used to infer configuration fields.
An MCore adapter explicitly forwards optimizer settings, while preserving the
existing FSDP owner gather/scatter and sharded momentum implementation.

DDP and MFSDP share a whole-matrix orthogonalizer. It supplies the r2s10 schedule
and normalization epsilon to Emerging Optimizers without changing its global
coefficient registry. FC1 splits rows into gate/up halves; MTP EH splits input
columns; fused QKV and indexer Q can split rows by head. QKV component splitting
retains precedence if both QKV and head options are set. The reference preset
explicitly disables QKV component splitting.

Tag logical parameter roles before creating optimizer masters. Copy those tags
into DDP FP32 masters. MFSDP tags its sharded Parameters after wrapping; owner
updates read these tags while operating on the assembled full matrix.

The additional kernel path requires TP=ETP=1 and no GTP weight rematerialization.
Existing default DDP algorithms retain their existing path. Layer-sharded Muon
rejects the new options rather than ignoring them. This change does not implement
vision-specific optimizer rules or checkpoint conversion.
