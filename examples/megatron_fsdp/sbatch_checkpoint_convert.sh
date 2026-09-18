#!/bin/bash

# Configuration: Set these paths before running the script
MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Path to Megatron-LM repository
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # Path to .sqsh or docker image url
OUTPUT_PATH=${OUTPUT_PATH:-"your_own_output_path"} # Path for SLURM logs

# Conversion options.
# MODEL_WEIGHTS_ONLY=1 (recommended) emits only model weights and no optimizer
# state. Keep it at 1 for Megatron-FSDP v2, which cannot save or load optimizer
# state. Set it to 0 only to also convert optimizer state for a consumer that
# supports it; that path also requires PARAM_TO_PARAM_GROUP_MAP_JSON below.
MODEL_WEIGHTS_ONLY=${MODEL_WEIGHTS_ONLY:-1}

# LOADABLE_LAYOUT=1 writes <out>/iter_<N>/ plus latest_checkpointed_iteration.txt,
# the layout Megatron's --load requires. Keep it at 1: without it, --load on the
# flat output directory only warns and silently starts from random weights.
LOADABLE_LAYOUT=${LOADABLE_LAYOUT:-1}

# SWIGLU=1 adds --swiglu and is required when the model config has swiglu: true
# (e.g. deepseek_v3_proxy). Set it to 0 for non-SwiGLU models.
SWIGLU=${SWIGLU:-1}

# Only used when MODEL_WEIGHTS_ONLY=0 and the source checkpoint has multiple
# optimizer param groups. Not needed for weights-only output.
PARAM_TO_PARAM_GROUP_MAP_JSON=${PARAM_TO_PARAM_GROUP_MAP_JSON:-""}

EXTRA_FLAGS=""
[ "${SWIGLU}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --swiglu"
[ "${MODEL_WEIGHTS_ONLY}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --model-weights-only"
[ "${LOADABLE_LAYOUT}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --loadable-layout"
[ "${MODEL_WEIGHTS_ONLY}" != "1" ] && [ -n "${PARAM_TO_PARAM_GROUP_MAP_JSON}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --param-to-param-group-map-json ${PARAM_TO_PARAM_GROUP_MAP_JSON}"

# Checkpoint conversion command
# Note: Update the checkpoint paths in the command below.
# The converter builds a CUDA DeviceMesh and initializes NCCL, so it must run in a
# GPU allocation, one process per GPU, with RANK/WORLD_SIZE/LOCAL_RANK set. For a
# local run use `torchrun --nproc_per_node=<GPUs>
# tools/checkpoint/checkpoint_inspector.py ...`; the SLURM launcher below is
# expected to provide the same per-process environment.
RUN_CMD="
cd ${MEGATRON_PATH};
git rev-parse HEAD;
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH};
python3 tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor${EXTRA_FLAGS} \
    your_own_path_to_input_torch_dist_checkpoint \
    your_own_path_to_output_fsdp_dtensor_checkpoint"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Submit SLURM job
# Note: Update SBATCH parameters below according to your cluster configuration
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=your_own_job_name
#SBATCH --partition=your_own_partition
#SBATCH --nodes=your_own_num_nodes
#SBATCH --ntasks-per-node=your_own_tasks_per_node
#SBATCH --gres=gpu:your_own_gpu_per_node
#SBATCH --time=your_own_time
#SBATCH --account=your_own_account
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=your_own_container_mounts \
    --container-workdir=${MEGATRON_PATH} \
    bash -x -c "${RUN_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e
