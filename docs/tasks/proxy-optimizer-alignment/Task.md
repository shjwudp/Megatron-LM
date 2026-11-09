# Proxy optimizer alignment

Make the proxy model's MFSDP v2 and DDP paths use the reference Adam/Muon
parameter policy and Muon algorithm. The reference uses r2s10, ten NS steps,
Nesterov momentum, head/FC1/EH splits and an explicit normalization epsilon.

Validate numerics and single-node training before requesting approval to rerun
the larger performance comparison. Keep workload presets and job submission
scripts outside the feature repository. New performance results require fresh runs.
