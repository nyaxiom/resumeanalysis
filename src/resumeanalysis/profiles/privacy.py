"""Suppress provider diagnostic details only within a private profile request."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar

_private_call = ContextVar("resumeanalysis_private_profile_call", default=False)


class _PrivateCallFilter(logging.Filter):
    def filter(self, record):
        return not _private_call.get()


_filter = _PrivateCallFilter()


@contextmanager
def private_provider_logs():
    # SDK construction imports its loggers before this context starts. Filters
    # remain inert outside this context, including in other concurrent threads.
    names = set(logging.Logger.manager.loggerDict) | {"instructor.v2.core.retry"}
    for name in names:
        if name.split(".")[0] in {"instructor", "openai", "httpx2", "httpcore2"}:
            logger = logging.getLogger(name)
            if _filter not in logger.filters:
                logger.addFilter(_filter)
    token = _private_call.set(True)
    try:
        yield
    finally:
        _private_call.reset(token)
