"""Tests for AdaptiveDecayCalculator — time-decay scoring for memory retrieval."""

import math
import pytest
from prot.decay import AdaptiveDecayCalculator, MEMORY_TYPE_MULTIPLIERS


class TestAdaptiveDecayCalculator:
    def setup_method(self):
        self.calc = AdaptiveDecayCalculator()

    def test_no_decay_at_zero_hours(self):
        result = self.calc.calculate(importance=0.8, hours_passed=0.0)
        assert abs(result - 0.8) < 1e-6

    def test_basic_decay(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100.0)
        expected = 1.0 * math.exp(-0.002 * 100.0)
        assert abs(result - expected) < 1e-6

    def test_minimum_retention_floor(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100000.0)
        assert result >= 1.0 * 0.1 - 1e-6

    def test_access_count_slows_decay(self):
        no_access = self.calc.calculate(importance=1.0, hours_passed=500.0, access_count=0)
        high_access = self.calc.calculate(importance=1.0, hours_passed=500.0, access_count=100)
        assert high_access > no_access

    def test_connection_count_slows_decay(self):
        no_conn = self.calc.calculate(importance=1.0, hours_passed=500.0, connection_count=0)
        high_conn = self.calc.calculate(importance=1.0, hours_passed=500.0, connection_count=10)
        assert high_conn > no_conn

    def test_fact_decays_slower_than_conversation(self):
        conv = self.calc.calculate(importance=1.0, hours_passed=500.0, memory_type="conversation")
        fact = self.calc.calculate(importance=1.0, hours_passed=500.0, memory_type="fact")
        assert fact > conv

    def test_recency_paradox_boost(self):
        # Old (>168h) but recently accessed (<24h)
        with_boost = self.calc.calculate(
            importance=1.0, hours_passed=200.0, last_access_hours=10.0
        )
        # Old (>168h) but NOT recently accessed (>24h)
        without_boost = self.calc.calculate(
            importance=1.0, hours_passed=200.0, last_access_hours=50.0
        )
        assert with_boost > without_boost

    def test_unknown_memory_type_uses_default(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100.0, memory_type="unknown")
        default = self.calc.calculate(importance=1.0, hours_passed=100.0, memory_type="conversation")
        assert abs(result - default) < 1e-6

    def test_memory_type_multipliers_defined(self):
        assert MEMORY_TYPE_MULTIPLIERS["fact"] == 0.3
        assert MEMORY_TYPE_MULTIPLIERS["preference"] == 0.5
        assert MEMORY_TYPE_MULTIPLIERS["insight"] == 0.7
        assert MEMORY_TYPE_MULTIPLIERS["conversation"] == 1.0

    def test_calculate_batch(self):
        memories = [
            {"importance": 0.8, "hours_passed": 50.0},
            {"importance": 1.0, "hours_passed": 200.0, "access_count": 10},
            {"importance": 0.5, "hours_passed": 0.0},
        ]
        results = self.calc.calculate_batch(memories)
        assert len(results) == 3
        assert abs(results[2] - 0.5) < 1e-6  # zero hours = no decay

    def test_calculate_batch_empty(self):
        assert self.calc.calculate_batch([]) == []

    def test_custom_config(self):
        calc = AdaptiveDecayCalculator(base_rate=0.01, min_retention=0.2)
        result = calc.calculate(importance=1.0, hours_passed=100000.0)
        assert result >= 1.0 * 0.2 - 1e-6
