export PYTHONPATH=/home/xueh/workspace/projects/multimodal/megatron-lm-m4:$PYTHONPATH

DATETIME=`date +'date_%y%m%d_%H%M'`

VISION_TP=$1
VISION_PP=$2
VISION_DP=$3
LANGUAGE_TP=$4
LANGUAGE_PP=$5
LANGUAGE_DP=$6
FSDP=$7
ENABLE_PROFILING=${8:-0}

TOTAL_GPUS=$((VISION_TP * VISION_PP * VISION_DP + LANGUAGE_TP * LANGUAGE_PP * LANGUAGE_DP))

PREFIX="etp${VISION_TP}epp${VISION_PP}edp${VISION_DP}dtp${LANGUAGE_TP}dpp${LANGUAGE_PP}ddp${LANGUAGE_DP}_fsdp${FSDP}"

if [ $ENABLE_PROFILING -eq 1 ]; then
    nsys profile -s none -t nvtx,cuda --cudabacktrace=all --cuda-graph-trace=node --python-backtrace=cuda -o heterogenous_parallel_${PREFIX}_${DATETIME} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nproc-per-node ${TOTAL_GPUS} train.py --vision_tp ${VISION_TP} --vision_pp ${VISION_PP} --vision_dp ${VISION_DP} --language_tp ${LANGUAGE_TP} --language_pp ${LANGUAGE_PP} --language_dp ${LANGUAGE_DP} 2>&1|tee out_${PREFIX}.log
else
  if [ $FSDP -eq 1 ]; then
    torchrun --nproc-per-node ${TOTAL_GPUS} train.py --vision_tp ${VISION_TP} --vision_pp ${VISION_PP} --vision_dp ${VISION_DP} --language_tp ${LANGUAGE_TP} --language_pp ${LANGUAGE_PP} --language_dp ${LANGUAGE_DP} --use_megatron_fsdp 2>&1|tee out_${PREFIX}.log
  else
    torchrun --nproc-per-node ${TOTAL_GPUS} train.py --vision_tp ${VISION_TP} --vision_pp ${VISION_PP} --vision_dp ${VISION_DP} --language_tp ${LANGUAGE_TP} --language_pp ${LANGUAGE_PP} --language_dp ${LANGUAGE_DP} 2>&1|tee out_${PREFIX}.log
  fi
fi
