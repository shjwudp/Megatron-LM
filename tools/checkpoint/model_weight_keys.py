# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure key-layout helpers for the ``torch_dist`` -> ``fsdp_dtensor`` converter.

This module deliberately imports nothing but ``re`` so the model-weight key
layout can be unit tested on a CPU-only machine; ``checkpoint_inspector.py``
(the torch/CUDA entry point) imports these helpers.

\b
Source layout
=============
``generate_state_dict`` writes a single top-level ``model`` section for one model
chunk and ``model0``, ``model1``, ... (one per chunk, in chunk order) under
virtual pipeline parallelism. The converter's source keys therefore are either
bare parameter names (single chunk) or carry a ``model{i}.`` section prefix::

    decoder.layers.0.mlp.linear_fc1.weight          # single chunk
    model0.decoder.layers.0.mlp.linear_fc1.weight   # VPP, chunk 0
    model1.decoder.layers.4.mlp.linear_fc1.weight   # VPP, chunk 1

\b
Destination layout
==================
The ``fsdp_dtensor`` loader requests model weights under the wrapper namespace,
section by section: ``model.module.<param>`` for a single chunk and
``model{i}.module.<param>`` under VPP. ``--output-model-weight-prefix``
(``model.module`` by default) supplies that wrapper namespace; sectioned input
replaces the prefix's leading ``model`` component with the actual section, so the
wrapper suffix (``module``) is preserved.

.. note::
   The VPP source layout above is derived from ``generate_state_dict`` and from
   the loader's request construction. It has **not** been confirmed against a
   real VPP ``torch_dist`` checkpoint (that requires a GPU run), so detection is
   explicit and conservative: only keys matching ``^model\\d+\\.`` are treated as
   sectioned, and everything else keeps the pre-existing single-section
   behaviour unchanged.
"""

import re

# Sectioned source key: ``model0.``, ``model1.``, ... (the same idea as
# ``fsdp_dtensor_checkpoint._MODEL_SECTION_PATTERN``).
MODEL_SECTION_PATTERN = re.compile(r'^model(\d+)\.')


def split_model_section(key):
    """Split ``model{i}.<rest>`` into ``('model{i}', '<rest>')``.

    Returns ``(None, key)`` for a key without a section prefix, so callers can
    branch on the section only.
    """
    match = MODEL_SECTION_PATTERN.match(key)
    if match is None:
        return None, key
    # Normalize the index (`model007` -> `model7`) so it can index the chunk list.
    section = f'model{int(match.group(1))}'
    return section, key[match.end() :]


def sectioned_model_weight_prefix(section, output_model_weight_prefix):
    """Output key prefix for one VPP model section.

    ``--output-model-weight-prefix`` describes the wrapper namespace
    (``model.module`` by default: the section root ``model`` plus the
    Megatron-FSDP ``module.`` level). For a sectioned source key the section must
    lead the on-disk key -- that is the section the ``fsdp_dtensor`` loader
    requests -- so a leading ``model`` component of the configured prefix is
    replaced by the section and the remaining components are kept::

        ('model0', 'model.module') -> 'model0.module'
        ('model1', 'model.module') -> 'model1.module'

    A prefix without a leading ``model`` component is appended after the section
    (``('model0', 'wrapped') -> 'model0.wrapped'``), so a custom
    ``--output-model-weight-prefix`` still chooses the wrapper namespace.
    """
    head, _, tail = output_model_weight_prefix.partition('.')
    if head == 'model':
        return section if not tail else f'{section}.{tail}'
    return f'{section}.{output_model_weight_prefix}'


def model_weight_output_key(source_key, output_model_weight_prefix='model.module'):
    """Map a source ``torch_dist`` model-weight key to its ``fsdp_dtensor`` output key.

    ``model{i}.<param>`` becomes ``model{i}.module.<param>`` (the section index is
    preserved) and a bare ``<param>`` becomes ``model.module.<param>`` exactly as
    before this helper existed. The wrapper component comes from
    ``output_model_weight_prefix`` as described in
    :func:`sectioned_model_weight_prefix`.
    """
    section, rest = split_model_section(source_key)
    if section is None:
        return f'{output_model_weight_prefix}.{source_key}'
    return f'{sectioned_model_weight_prefix(section, output_model_weight_prefix)}.{rest}'
