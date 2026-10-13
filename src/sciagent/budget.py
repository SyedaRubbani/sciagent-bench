"""Budget accounting, independent of any planner.

Compute is handled with reserve -> commit/release, so the invariant ``committed + reserved <= total``
holds at every instant, whatever a planner (or an LLM) proposes, and even while a batch is executing.
LLM spend is bounded *before* each call using a worst-case estimate (prompt length plus ``max_tokens``),
so a call that could break the cap is never made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EPS = 1e-9


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class _Reservation:
    token: int
    cost: float
    ref: str


@dataclass
class BudgetLedger:
    total: float
    committed: float = 0.0
    reserved: float = 0.0
    entries: list[dict] = field(default_factory=list)
    _open: dict[int, _Reservation] = field(default_factory=dict)
    _next: int = 0

    @property
    def remaining(self) -> float:
        return self.total - self.committed - self.reserved

    def can_afford(self, cost: float) -> bool:
        return cost <= self.remaining + EPS

    def reserve(self, cost: float, ref: str = "") -> int:
        if cost < 0:
            raise ValueError("negative cost")
        if not self.can_afford(cost):
            raise BudgetExceeded(f"cannot reserve {cost:g}: remaining {self.remaining:.6g} of {self.total:g}")
        self._next += 1
        self._open[self._next] = _Reservation(self._next, cost, ref)
        self.reserved += cost
        return self._next

    def commit(self, token: int, cost: float | None = None, ref: str = "") -> float:
        """Charge a reservation; ``cost`` may be lower than reserved (e.g. a cache hit costs 0), never higher."""
        r = self._open.pop(token)
        self.reserved -= r.cost
        c = r.cost if cost is None else float(cost)
        if c > r.cost + EPS:
            raise BudgetExceeded(f"commit {c:g} exceeds reservation {r.cost:g}")
        self.committed += c
        self.entries.append({"ref": ref or r.ref, "reserved": r.cost, "charged": c})
        return c

    def release(self, token: int) -> None:
        r = self._open.pop(token, None)
        if r:
            self.reserved -= r.cost

    def release_all(self) -> None:
        for t in list(self._open):
            self.release(t)

    def set_total(self, total: float) -> float:
        """Researcher changes the budget; it can never drop below what is already spent or reserved."""
        self.total = max(float(total), self.committed + self.reserved)
        return self.total

    def check(self) -> None:
        assert self.committed + self.reserved <= self.total + 1e-6, "budget invariant violated"
        assert self.reserved >= -1e-9 and self.committed >= -1e-9


def llm_call_affordable(spent_usd: float, cap_usd: float | None, prompt_chars: int, max_tokens: int,
                        price_in_per_mtok: float, price_out_per_mtok: float) -> tuple[bool, float]:
    """Worst-case USD of one call (~4 chars/token for the prompt, max_tokens for the reply)."""
    worst = (prompt_chars / 3.0 * price_in_per_mtok + max_tokens * price_out_per_mtok) / 1e6
    if cap_usd is None:
        return True, worst
    return spent_usd + worst <= cap_usd + EPS, worst
