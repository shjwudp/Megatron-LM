# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Importability of the MFSDP v2 combined-1F1B wiring.

``megatron/core/models/common/combined_1f1b_mfsdp_scheduler.py`` is imported from
``megatron/core/distributed/fsdp/mcore_fsdp_adapter.py``, which
``megatron/core/optimizer/__init__.py`` pulls in while ``megatron/core/__init__.py``
is still executing -- before it binds ``InferenceParams``. A module-level import of
``megatron.core.transformer.multi_token_prediction`` from the scheduler closes a
cycle back to that partially initialized package and breaks ``import
megatron.core`` for the whole repository, at pytest COLLECTION time.

Two failure causes must stay separable, because they need opposite responses:

* an import **cycle** is a code bug and must fail, loudly, naming the offending
  import;
* a host whose ``torch`` predates the Megatron stack (this repository requires
  ``torch.float8_e8m0fnu``, which torch 2.4 does not define) cannot import the
  package for reasons that have nothing to do with this code.

So the dynamic import tests below are gated on a PRECONDITION evaluated before the
import -- ``torch.float8_e8m0fnu`` must exist -- and never on an
``except ImportError`` around it. The invariant:

    the skip condition must be a precondition evaluated before the import; never an
    except-ImportError, which would mask a cycle.

It is enforced here, not merely documented: the predicate is answerable without
importing Megatron, and
:func:`test_the_dynamic_import_skip_is_a_precondition_not_an_exception_handler`
asserts that property. The unconditional guard is
:func:`test_scheduler_has_no_module_level_import_of_multi_token_prediction`: static
(AST, no torch, no Megatron), so it runs and cannot skip on any host, and it names
the offending line when it fails.
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
# The environmental precondition: the oldest torch this tree's Megatron stack can
# import at all still needs this dtype, which torch 2.4 does not define.
_REQUIRED_TORCH_ATTR = "float8_e8m0fnu"

_SKIP_REASON = (
    f"host precondition unmet: torch has no '{_REQUIRED_TORCH_ATTR}' (added in torch "
    "2.8), so this host cannot import the Megatron stack for reasons unrelated to the "
    "import graph under test. The unconditional AST guard still runs here."
)


def _host_can_import_megatron_stack() -> bool:
    """Return whether this host satisfies the precondition for the dynamic imports.

    Evaluated BEFORE any Megatron import and answerable without touching the import
    graph: only the environmental cause (a missing torch attribute) is tested, so a
    cycle cannot reach this function and cannot turn into a skip. By the time an
    import runs, the skip decision has already been made on a fact that has nothing
    to do with that import.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency here
        return False
    return hasattr(torch, _REQUIRED_TORCH_ATTR)


def test_scheduler_has_no_module_level_import_of_multi_token_prediction():
    """Pin the exact import that closes the cycle, statically.

    This runs on any host -- it imports neither torch nor Megatron -- so it cannot
    skip, and it names the offending statement rather than failing later inside an
    unrelated collection error.
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


def test_the_import_skip_is_a_precondition_not_an_exception_handler():
    """Keep the gate honest: the skip must not be able to swallow a cycle.

    Checked statically, so it holds regardless of test order or host:

    * the precondition is answerable without importing Megatron (it only inspects
      ``torch``), so a cycle cannot influence the decision;
    * the dynamic test is gated by a ``skipif`` MARKER -- evaluated before the test
      body runs -- and its body contains no ``except ImportError`` that could turn a
      cycle into a skip.
    """
    # The predicate must agree with the environmental fact it claims to test.
    import torch

    assert _host_can_import_megatron_stack() == hasattr(torch, _REQUIRED_TORCH_ATTR)

    # The gate must be a marker, and the body must not catch import failures.
    tree = ast.parse(Path(__file__).read_text())
    gated = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "test_module_imports_without_a_cycle"
    ]
    assert gated, "test_module_imports_without_a_cycle is missing"

    decorators = [ast.unparse(decorator) for decorator in gated[0].decorator_list]
    assert any(
        "skipif" in decorator and "_host_can_import_megatron_stack()" in decorator
        for decorator in decorators
    ), (
        "the dynamic import test must be gated by a skipif marker whose condition is the "
        f"host precondition, so the skip is decided before the import. Decorators: {decorators}"
    )

    import_handlers = [
        ast.unparse(node)
        for node in ast.walk(gated[0])
        if isinstance(node, ast.ExceptHandler)
        and "ImportError" in ast.unparse(node.type)
        and "skip" in ast.unparse(node)
    ]
    assert not import_handlers, (
        "the dynamic import test must not convert an ImportError into a skip: that is "
        "exactly what would let an import cycle hide. Found: " + "; ".join(import_handlers)
    )


@pytest.mark.skipif(not _host_can_import_megatron_stack(), reason=_SKIP_REASON)
@pytest.mark.parametrize("module_name", ["megatron.core", _SCHEDULER_MODULE])
def test_module_imports_without_a_cycle(module_name):
    """Importing the module must not raise, and must not silently skip.

    The skip decision above was made on the host's torch, before this import ran, so
    on a host that satisfies the precondition a failure here is always a real
    failure: the body uses ``pytest.fail`` and never a skip marker.
    """
    try:
        module = importlib.import_module(module_name)
    except (ImportError, AttributeError) as error:
        pytest.fail(
            f"importing {module_name!r} raised {type(error).__name__}: {error}. This "
            f"host satisfies the {_REQUIRED_TORCH_ATTR} precondition, so the "
            f"environmental cause is excluded: it usually means a module-level import of "
            f"{_CYCLE_MODULE} closed the cycle described in "
            f"test_scheduler_has_no_module_level_import_of_multi_token_prediction, which "
            f"runs everywhere and names the offending line."
        )

    assert module.__name__ == module_name
