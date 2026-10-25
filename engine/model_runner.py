"""
Stand-in for the actual neural network forward pass.

A real engine would run transformer layers on a GPU here. This project
is about *scheduling and memory mechanics*, not training/loading a
real model, so this fakes "inference" with a bit of artificial latency
and a deterministic-ish next token.

It sits behind an abstract interface specifically so you could swap in
a real HF/transformers runner later without touching the scheduler or
cache code at all - that's the point of the abstraction.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Dict

from engine.scheduler import SchedulerOutput


class BaseModelRunner(ABC):
    @abstractmethod
    def forward(self, batch: SchedulerOutput) -> Dict[int, int]:
        """Run the batch, return {request_id: next_token_id} for decode steps only."""
        raise NotImplementedError


class MockModelRunner(BaseModelRunner):
    def __init__(self, eos_token_id: int, eos_probability: float = 0.15, simulate_latency: bool = False):
        self.eos_token_id = eos_token_id
        self.eos_probability = eos_probability
        self.simulate_latency = simulate_latency
        self._rng = random.Random(0)  # fixed seed - keeps demo runs reproducible

    def forward(self, batch: SchedulerOutput) -> Dict[int, int]:
        if self.simulate_latency:
            # Compute roughly scales with total tokens in the batch - this is
            # what makes a step with a big prefill chunk "cost" more than a
            # step of pure 1-token decodes, just like on real hardware.
            total_tokens = sum(item.num_tokens for item in batch.scheduled)
            time.sleep(0.001 * total_tokens)

        next_tokens: Dict[int, int] = {}
        for item in batch.scheduled:
            if item.is_prefill:
                continue  # a prefill chunk is just digesting prompt tokens, no output yet
            next_tokens[item.request.request_id] = self._sample_next_token()
        return next_tokens

    def _sample_next_token(self) -> int:
        if self._rng.random() < self.eos_probability:
            return self.eos_token_id
        return self._rng.randint(0, 50_000)
