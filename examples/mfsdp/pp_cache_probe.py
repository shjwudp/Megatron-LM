# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Instrument a standalone MFSDP PP experiment; run with normal pretrain_hybrid flags.

PP_CACHE_OUTPUT selects the output directory. PP_CACHE_CHECK=1 saves every local
parameter/gradient shard after each update for matching cached/uncached runs.
Instrumentation counts logical collective payload, not physical network bytes.
"""

import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import megatron.training.training as training
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)

counts = Counter()
original_gather = FsdpParameterGroup._materialize_unsharded_model_weight
original_reduce = FsdpParameterGroup.reduce_partial_gradients
original_train_step = training.train_step
step = 0


def gathered(group, **planes):
    buffer = group._unsharded_model_weight
    buffers = (buffer,) if isinstance(buffer, DBuffer) else buffer.planes_for(**planes)
    counts['gather_calls'] += 1
    counts['gather_full_payload_bytes'] += sum(
        item.local_buffer.numel() * item.local_buffer.element_size() for item in buffers
    )
    return original_gather(group, **planes)


def reduced(group, partial_grad, **kwargs):
    counts['reduce_calls'] += 1
    counts['reduce_input_payload_bytes'] += (
        partial_grad.local_buffer.numel() * partial_grad.local_buffer.element_size()
    )
    return original_reduce(group, partial_grad, **kwargs)


def train_step(*args, **kwargs):
    global step
    counts.clear()
    result = original_train_step(*args, **kwargs)
    step += 1
    rank = torch.distributed.get_rank()
    output = Path(os.environ['PP_CACHE_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    record = dict(counts, step=step, rank=rank,
                  allocated_bytes=torch.cuda.max_memory_allocated(),
                  reserved_bytes=torch.cuda.max_memory_reserved())
    with (output / f'rank-{rank}.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    if os.environ.get('PP_CACHE_CHECK') == '1':
        models = args[2] if len(args) > 2 else kwargs['model']
        state = {}
        for chunk_id, model in enumerate(models):
            for name, parameter in model.named_parameters():
                local = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
                grad = parameter.grad
                if grad is not None:
                    grad = grad.to_local() if hasattr(grad, 'to_local') else grad
                state[f'{chunk_id}:{name}'] = (
                    local.detach().cpu().clone(),
                    None if grad is None else grad.detach().cpu().clone(),
                )
        torch.save(state, output / f'step-{step}-rank-{rank}.pt')
    return result


FsdpParameterGroup._materialize_unsharded_model_weight = gathered
FsdpParameterGroup.reduce_partial_gradients = reduced
training.train_step = train_step

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    if runtime_hook := os.environ.get('PP_CACHE_RUNTIME_HOOK'):
        runpy.run_path(runtime_hook, run_name='__main__')
    runpy.run_path(str(root / 'pretrain_hybrid.py'), run_name='__main__')
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
