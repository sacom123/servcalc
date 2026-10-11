"""Request shape: prompt and generation lengths plus the concurrency sweep."""

from __future__ import annotations

from dataclasses import dataclass


def _positive_int(name: str, v) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")


@dataclass(frozen=True)
class Workload:
    prompt_len: int
    gen_len: int
    concurrency: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    def __post_init__(self):
        _positive_int("prompt_len", self.prompt_len)
        if not isinstance(self.gen_len, int) or isinstance(self.gen_len, bool) or self.gen_len < 2:
            raise ValueError(
                "gen_len must be an integer >= 2 (prefill emits the first token), "
                f"got {self.gen_len!r}")
        if not self.concurrency:
            raise ValueError("concurrency must contain at least one value")
        for n in self.concurrency:
            _positive_int("concurrency", n)
        if list(self.concurrency) != sorted(set(self.concurrency)):
            raise ValueError("concurrency must be strictly ascending")

    @property
    def decode_steps(self) -> int:
        return self.gen_len - 1

    @property
    def mean_decode_ctx(self) -> float:
        """Average context during decode: step j has I + 1 + j for 0 <= j < D."""
        return self.prompt_len + 1 + (self.decode_steps - 1) / 2
