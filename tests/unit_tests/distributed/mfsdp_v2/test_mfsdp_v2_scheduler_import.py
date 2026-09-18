# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Importability of the MFSDP v2 combined-1F1B wiring.

``megatron/core/models/common/combined_1f1b_mfsdp_scheduler.py`` is imported from
``megatron/core/distributed/fsdp/mcore_fsdp_adapter.py``, which
``megatron/core/optimizer/__init__.py`` pulls in while ``megatron/core/__init__.py``
is still executing -- before it binds ``InferenceParams``. A module-level import of
``megatron.core.transformer.multi_token_prediction`` from the scheduler closes a
cycle back to that partially initialized package and breaks ``import
megatron.core`` for the whole repository, at pytest COLLECTION time.

These tests assert the import succeeds and are deliberately written so they cannot
degrade into a skip. The other MFSDP v2 test file routes imports through
``_import_or_skip`` because the GPU stack is legitimately optional; this file does
not, because an import cycle is never environmental.
"""

import ast
import importlib
from pathlib import Path

import pytest

_SCHEDULER_MODULE = "megatron.core.models.common.combined_1f1b_mfsdp_scheduler"
_SCHEDULER_PATH = (
    Path(__file__).resolve().parents[4]
    / "megatron"
    / "core"
    / "models"
    / "common"
    / "combined_1f1b_mfsdp_scheduler.py"
)
# The module whose import closes the cycle, because it does
# ``from megatron.core import InferenceParams`` at module scope.
_CYCLE_MODULE = "megatron.core.transformer.multi_token_prediction"


def test_scheduler_has_no_module_level_import_of_multi_token_prediction():
    """Pin the exact import that closes the cycle, statically.

    This runs on any host -- it does not import torch or Megatron at all -- so it
    cannot skip, and it names the offending statement rather than failing later
    inside an unrelated collection error.
    """
    tree = ast.parse(_SCHEDULER_PATH.read_text())

    offenders = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == _CYCLE_MODULE:
            offenders.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            offenders.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name == _CYCLE_MODULE
            )

    assert not offenders, (
        "combined_1f1b_mfsdp_scheduler.py must not import "
        f"{_CYCLE_MODULE} at module scope: it is imported from mcore_fsdp_adapter while "
        "megatron/core/__init__.py is still initializing, so that import closes the cycle "
        f"mcore_fsdp_adapter -> scheduler -> multi_token_prediction -> 'from megatron.core "
        f"import InferenceParams'. Import it lazily inside the function that needs it, or "
        f"derive the value without it. Found: {offenders}"
    )


@pytest.mark.parametrize("module_name", ["megatron.core", _SCHEDULER_MODULE])
def test_module_imports_without_a_cycle(module_name):
    """Importing the module must not raise, and must not silently skip.

    ``pytest.fail`` rather than a skip marker, so a broken import chain is red on
    every host. A host whose torch predates the Megatron stack fails here too: that
    is reported as a failure naming the host limitation, not excused as an
    environmental skip, because a genuine import cycle can raise the same exception
    types and must not be able to hide behind one.
    """
    try:
        module = importlib.import_module(module_name)
    except (ImportError, AttributeError) as error:
        pytest.fail(
            f"importing {module_name!r} raised {type(error).__name__}: {error}. If this "
            f"names a missing torch attribute (for example torch.float8_e8m0fnu) the host's "
            f"torch is older than this repository requires, and the failure is environmental "
            f"rather than a cycle. Otherwise it usually means a module-level import of "
            f"{_CYCLE_MODULE} closed the cycle described in "
            f"test_scheduler_has_no_module_level_import_of_multi_token_prediction — check "
            f"that test first, it runs everywhere and names the offending line."
        )

    assert module.__name__ == module_name
