"""Adaptive decay calculator for memory importance scoring.

Ported from genuine-axel's AdaptiveDecayCalculator, simplified:
- No channel_mentions / channel_diversity
- No circadian stability / dynamic_decay
- No native C++ module
- Pure Python only
"""

from __future__ import annotations

import math

# Memory type-specific decay multipliers (lower = slower decay)
MEMORY_TYPE_MULTIPLIERS: dict[str, float] = {
    "fact": 0.3,
    "preference": 0.5,
    "insight": 0.7,
    "conversation": 1.0,
}

# Recency paradox thresholds
_RECENCY_AGE_HOURS = 168       # 1 week
_RECENCY_ACCESS_HOURS = 24     # accessed within 24h
_RECENCY_BOOST = 1.3

# Stability/resistance constants
_ACCESS_STABILITY_K = 0.3
_RELATION_RESISTANCE_K = 0.1


class AdaptiveDecayCalculator:
    """Time-decay scoring for memory retrieval.

    Core formula:
        effective_rate = base_rate * type_mult / stability * (1 - resistance)
        decayed = importance * exp(-effective_rate * hours_passed)

    Modulating factors:
        - stability: 1 + 0.3 * log(1 + access_count) — more access = slower decay
        - resistance: min(1.0, connection_count * 0.1) — more connections = slower decay
        - type_multiplier: fact(0.3) < preference(0.5) < insight(0.7) < conversation(1.0)
        - recency boost: old but recently accessed → 1.3x
        - min retention floor: never below importance * min_retention
    """

    def __init__(
        self,
        base_rate: float = 0.002,
        min_retention: float = 0.1,
    ) -> None:
        self.base_rate = base_rate
        self.min_retention = min_retention

    def calculate(
        self,
        importance: float,
        hours_passed: float,
        access_count: int = 0,
        connection_count: int = 0,
        last_access_hours: float = -1.0,
        memory_type: str = "conversation",
    ) -> float:
        """Calculate decayed importance score."""
        stability = 1 + _ACCESS_STABILITY_K * math.log(1 + access_count)
        resistance = min(1.0, connection_count * _RELATION_RESISTANCE_K)
        type_mult = MEMORY_TYPE_MULTIPLIERS.get(memory_type, 1.0)

        effective_rate = self.base_rate * type_mult / stability * (1 - resistance)
        decayed = importance * math.exp(-effective_rate * hours_passed)

        # Recency paradox: old memory recently accessed gets boost
        if (
            last_access_hours >= 0
            and hours_passed > _RECENCY_AGE_HOURS
            and last_access_hours < _RECENCY_ACCESS_HOURS
        ):
            decayed *= _RECENCY_BOOST

        return max(decayed, importance * self.min_retention)

    def calculate_batch(self, memories: list[dict]) -> list[float]:
        """Calculate decayed importance for a batch of memories."""
        return [
            self.calculate(
                importance=m.get("importance", 0.5),
                hours_passed=m.get("hours_passed", 0.0),
                access_count=m.get("access_count", 0),
                connection_count=m.get("connection_count", 0),
                last_access_hours=m.get("last_access_hours", -1.0),
                memory_type=m.get("memory_type", "conversation"),
            )
            for m in memories
        ]
