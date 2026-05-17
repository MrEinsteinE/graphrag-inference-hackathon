from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineResult:
    name: str
    answer: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def cost_usd(self, price_in: float, price_out: float) -> float | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        pin = (self.prompt_tokens or 0) / 1_000_000 * price_in
        pout = (self.completion_tokens or 0) / 1_000_000 * price_out
        return round(pin + pout, 6)


@dataclass
class AccuracyResult:
    judge_pass: bool | None
    judge_raw: str
    bertscore_f1: float | None
    bertscore_precision: float | None
    bertscore_recall: float | None
    error: str | None = None
