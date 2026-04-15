"""Round-robin fleet-level router.

Cycles through pools in order, distributing requests evenly regardless of
pool capacity or request characteristics.  Useful as a baseline.
"""

from __future__ import annotations

import itertools

from ..core.fleet import PoolConfig
from ..core.request import Request
from .base import BaseRouter


class RoundRobinRouter(BaseRouter):
    def __init__(self, pools: dict[str, PoolConfig], **kwargs):
        super().__init__(pools, **kwargs)
        self._cycle = itertools.cycle(self.pool_ids)

    def route(self, req: Request) -> str | None:
        return next(self._cycle)
