"""
Utility to inspect what tensors are saved on the autograd tape after forward.

Usage: After capture_forward(), call inspect_autograd_tape(runner._static_outputs)
to see exactly what saved tensors the backward will need.

This tells you what's consuming the ~4GB across 60 layers and helps you
decide what to eliminate (e.g., via activation checkpointing on specific ops).
"""

import torch
from collections import defaultdict


def inspect_autograd_tape(outputs, max_depth=50):
    """Walk the autograd graph from outputs and collect all saved tensors.

    Returns a list of dicts with:
      - size: tensor size in bytes
      - shape: tensor shape
      - dtype: tensor dtype
      - grad_fn_name: which grad_fn saved this tensor
      - depth: how deep in the autograd graph
    """
    visited_fns = set()
    saved_info = []

    def _walk(grad_fn, depth=0):
        if grad_fn is None or id(grad_fn) in visited_fns or depth > max_depth:
            return
        visited_fns.add(id(grad_fn))

        fn_name = type(grad_fn).__name__

        # Access saved tensors via the internal _saved_tensors or
        # the public saved_tensors property (if available)
        try:
            saved = grad_fn._saved_tensors if hasattr(grad_fn, '_saved_tensors') else []
        except Exception:
            saved = []

        # Alternative: try the newer saved_tensors() method
        if not saved:
            try:
                # In newer PyTorch, grad_fn has .saved_tensors attribute
                for attr_name in dir(grad_fn):
                    if attr_name.startswith('_saved_') and not attr_name.startswith('_saved_tensors'):
                        val = getattr(grad_fn, attr_name, None)
                        if isinstance(val, torch.Tensor) and val.is_cuda:
                            saved_info.append({
                                'size_bytes': val.nelement() * val.element_size(),
                                'shape': tuple(val.shape),
                                'dtype': str(val.dtype),
                                'grad_fn_name': fn_name,
                                'attr_name': attr_name,
                                'depth': depth,
                                'data_ptr': val.data_ptr(),
                            })
            except Exception:
                pass

        for t in saved:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                saved_info.append({
                    'size_bytes': t.nelement() * t.element_size(),
                    'shape': tuple(t.shape),
                    'dtype': str(t.dtype),
                    'grad_fn_name': fn_name,
                    'attr_name': '_saved_tensors',
                    'depth': depth,
                    'data_ptr': t.data_ptr(),
                })

        # Recurse into next_functions
        for next_fn, _ in grad_fn.next_functions:
            _walk(next_fn, depth + 1)

    if isinstance(outputs, (list, tuple)):
        for o in outputs:
            if isinstance(o, torch.Tensor) and o.grad_fn is not None:
                _walk(o.grad_fn)
    elif isinstance(outputs, torch.Tensor) and outputs.grad_fn is not None:
        _walk(outputs.grad_fn)

    return saved_info


def summarize_tape(saved_info):
    """Print a summary of the autograd tape contents."""
    if not saved_info:
        print("No saved tensors found on the autograd tape.")
        return

    total_bytes = sum(s['size_bytes'] for s in saved_info)
    print(f"\n{'='*80}")
    print(f"AUTOGRAD TAPE SUMMARY")
    print(f"{'='*80}")
    print(f"Total saved tensors: {len(saved_info)}")
    print(f"Total memory: {total_bytes / 1e6:.2f} MB ({total_bytes / 1e9:.4f} GB)")
    print()

    # Group by grad_fn_name
    by_fn = defaultdict(lambda: {'count': 0, 'total_bytes': 0, 'shapes': []})
    for s in saved_info:
        key = s['grad_fn_name']
        by_fn[key]['count'] += 1
        by_fn[key]['total_bytes'] += s['size_bytes']
        by_fn[key]['shapes'].append(s['shape'])

    print(f"{'Grad Function':<40} {'Count':>6} {'Total MB':>10} {'Shapes'}")
    print(f"{'-'*40} {'-'*6} {'-'*10} {'-'*30}")
    for fn_name, info in sorted(by_fn.items(), key=lambda x: -x[1]['total_bytes']):
        unique_shapes = list(set(info['shapes']))[:3]
        shapes_str = ', '.join(str(s) for s in unique_shapes)
        if len(set(info['shapes'])) > 3:
            shapes_str += ', ...'
        print(f"{fn_name:<40} {info['count']:>6} {info['total_bytes']/1e6:>10.2f} {shapes_str}")

    # Group by attr_name to show what kind of saved data
    print(f"\n{'Attribute':<40} {'Count':>6} {'Total MB':>10}")
    print(f"{'-'*40} {'-'*6} {'-'*10}")
    by_attr = defaultdict(lambda: {'count': 0, 'total_bytes': 0})
    for s in saved_info:
        key = f"{s['grad_fn_name']}.{s.get('attr_name', '?')}"
        by_attr[key]['count'] += 1
        by_attr[key]['total_bytes'] += s['size_bytes']
    for attr, info in sorted(by_attr.items(), key=lambda x: -x[1]['total_bytes']):
        print(f"{attr:<40} {info['count']:>6} {info['total_bytes']/1e6:>10.2f}")

    print(f"\n{'='*80}")


def inspect_tape_with_hooks(module, sample_inputs):
    """Use saved_tensors_hooks to precisely track what gets saved.

    This is the MOST RELIABLE method -- it intercepts every pack() call
    during forward, capturing the tensor and the call stack.

    Usage:
        saved = inspect_tape_with_hooks(model.layer_0, (input_tensor,))
        summarize_hook_results(saved)
    """
    import traceback

    saved_tensors = []

    def pack_hook(tensor):
        # Capture the tensor info and the call stack at save time
        saved_tensors.append({
            'size_bytes': tensor.nelement() * tensor.element_size(),
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'data_ptr': tensor.data_ptr(),
            'stack': traceback.format_stack(limit=10),
        })
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        if isinstance(sample_inputs, (list, tuple)):
            out = module(*sample_inputs)
        else:
            out = module(sample_inputs)

    return saved_tensors, out


def summarize_hook_results(saved_tensors):
    """Print summary of tensors captured via saved_tensors_hooks."""
    if not saved_tensors:
        print("No saved tensors captured.")
        return

    total_bytes = sum(s['size_bytes'] for s in saved_tensors)
    print(f"\n{'='*80}")
    print(f"SAVED TENSORS HOOKS SUMMARY")
    print(f"{'='*80}")
    print(f"Total saved tensors: {len(saved_tensors)}")
    print(f"Total memory: {total_bytes / 1e6:.2f} MB ({total_bytes / 1e9:.4f} GB)")
    print()

    # Group by the calling location (last meaningful frame in stack)
    by_location = defaultdict(lambda: {'count': 0, 'total_bytes': 0, 'shapes': set()})
    for s in saved_tensors:
        # Find the most meaningful frame (skip torch internals)
        location = "unknown"
        for frame in reversed(s['stack']):
            if '/torch/' not in frame and '/autograd/' not in frame and 'inspect_tape' not in frame:
                # Extract just file:line:function
                lines = frame.strip().split('\n')
                if lines:
                    location = lines[0].strip()
                break
        by_location[location]['count'] += 1
        by_location[location]['total_bytes'] += s['size_bytes']
        by_location[location]['shapes'].add(s['shape'])

    print(f"{'Location':<70} {'Count':>5} {'MB':>8}")
    print(f"{'-'*70} {'-'*5} {'-'*8}")
    for loc, info in sorted(by_location.items(), key=lambda x: -x[1]['total_bytes']):
        loc_short = loc[-67:] if len(loc) > 67 else loc
        print(f"{loc_short:<70} {info['count']:>5} {info['total_bytes']/1e6:>8.2f}")
        for shape in sorted(info['shapes'], key=lambda s: -sum(s)):
            print(f"  {'':70} shape={shape}")

    print(f"\n{'='*80}")


# =============================================================================
# QUICK INTEGRATION: Add this to your capture_forward right after the capture
# =============================================================================

INTEGRATION_EXAMPLE = """
# In FSDPCudaGraphRunner.capture_forward(), after line 544:
#     self._static_outputs = tuple(self._flatten_output(out))

# ADD THIS to inspect what's on the tape:
from inspect_autograd_tape import inspect_autograd_tape, summarize_tape
saved_info = inspect_autograd_tape(self._static_outputs)
summarize_tape(saved_info)

# OR use the more reliable hook-based method during warmup:
from inspect_autograd_tape import inspect_tape_with_hooks, summarize_hook_results
saved, _ = inspect_tape_with_hooks(self._module, static_inputs)
summarize_hook_results(saved)
"""

if __name__ == "__main__":
    print(INTEGRATION_EXAMPLE)
