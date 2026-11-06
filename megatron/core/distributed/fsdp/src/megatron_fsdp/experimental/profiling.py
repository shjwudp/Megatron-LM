"""Optional stream-range observer supplied by an embedding application."""

import functools
from contextlib import nullcontext

_scope_factory = None


def set_profile_scope_factory(factory):
    """Install a name -> context manager factory without depending on MCore."""
    global _scope_factory
    _scope_factory = factory


def profile_scope(name):
    """Observe a region on its caller's current stream without adding dependencies."""
    return nullcontext() if _scope_factory is None else _scope_factory(name)


def profile_operation(name):
    """Observe synchronous-on-stream operations, including their buffer preparation."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if _scope_factory is None:
                return function(*args, **kwargs)
            with _scope_factory(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate
