"""
The scheduler is the brain of continuous batching: on every single
iteration it decides, token-by-token-budget, which requests get to run
and whether they're doing a prefill chunk or a decode step. Unlike
old-school static batching (wait for a full batch, run it, wait for
ALL of it to finish before starting the next one), sequences can join
and leave the running batch on every iteration here.

Scheduling order each step, in priority order:
  1. decode steps for requests that already finished their prefill
     (cheapest - 1 token each - so they go first and almost never starve)
  2. prefill continuation for requests already admitted but mid-prompt
  3. admitting brand new requests from the waiting queue

If the KV cache is full and an already-running request needs more
space, we preempt the most-recently-admitted running request to make
room (simple "recompute" preemption: the victim keeps its generated
tokens but has to redo its KV cache from scratch when re-admitted).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from config import EngineConfig
from engine.chunked_prefill import PrefillChunker
from engine.kv_cache import PagedKVCache
from engine.request import Request, RequestStatus


@dataclass
class ScheduledItem:
    request: Request
    num_tokens: int       # how many tokens we're feeding the model for this request, this step
    is_prefill: bool       # prefill chunk vs. a single decode token


@dataclass
class SchedulerOutput:
    scheduled: List[ScheduledItem] = field(default_factory=list)
    preempted: List[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class ContinuousBatchingScheduler:
    def __init__(self, config: EngineConfig, kv_cache: PagedKVCache):
        self.config = config
        self.kv_cache = kv_cache
        self.chunker = PrefillChunker(config.max_num_batched_tokens)

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    # -- the main event loop body --------------------------------------------

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        budget = self._step_decoding_requests(output, budget)
        budget = self._continue_prefilling_requests(output, budget)
        budget = self._admit_new_requests(output, budget)

        return output

    def _step_decoding_requests(self, output: SchedulerOutput, budget: int) -> int:
        """Priority #1: anyone already past prefill just needs 1 more token of space."""
        still_running: List[Request] = []

        for request in list(self.running):
            if request.status != RequestStatus.RUNNING:
                continue  # already preempted as someone else's victim this step
            if not request.is_prefill_done:
                still_running.append(request)  # not this pass's job
                continue
            if budget <= 0:
                still_running.append(request)
                continue

            if not self._ensure_cache_space(request, num_new_tokens=1, output=output):
                self._preempt(request, output)
                continue

            self.kv_cache.allocate_for_tokens(request.request_id, request.num_computed_tokens, 1)
            output.scheduled.append(ScheduledItem(request, num_tokens=1, is_prefill=False))
            budget -= 1
            still_running.append(request)

        self.running = still_running
        return budget

    def _continue_prefilling_requests(self, output: SchedulerOutput, budget: int) -> int:
        """Priority #2: give in-progress prefills another chunk, if the budget allows."""
        still_running: List[Request] = []

        for request in list(self.running):
            if request.status != RequestStatus.RUNNING or request.is_prefill_done:
                still_running.append(request)
                continue

            chunk_size = self.chunker.next_chunk_size(request.num_pending_prompt_tokens, budget)
            if chunk_size <= 0:
                still_running.append(request)  # budget's gone - try again next step
                continue

            if not self._ensure_cache_space(request, chunk_size, output=output):
                self._preempt(request, output)
                continue

            self.kv_cache.allocate_for_tokens(request.request_id, request.num_computed_tokens, chunk_size)
            output.scheduled.append(ScheduledItem(request, num_tokens=chunk_size, is_prefill=True))
            budget -= chunk_size
            still_running.append(request)

        self.running = still_running
        return budget

    def _admit_new_requests(self, output: SchedulerOutput, budget: int) -> int:
        """Priority #3: spend whatever's left of the budget pulling in fresh requests."""
        still_waiting: Deque[Request] = deque()

        while self.waiting:
            if budget <= 0 or len(self.running) >= self.config.max_num_seqs:
                still_waiting.extend(self.waiting)
                break

            request = self.waiting.popleft()
            chunk_size = self.chunker.next_chunk_size(request.num_pending_prompt_tokens, budget)

            # Note allow_preempt=False here: a brand new request should never
            # evict an already-running one just to get its first chunk in -
            # that would starve requests that are already making progress.
            if chunk_size <= 0 or not self._ensure_cache_space(request, chunk_size, allow_preempt=False):
                # This one didn't fit - put it back, along with everyone
                # still behind it in the queue (they haven't even been
                # looked at yet, so they must go back too, not vanish).
                still_waiting.append(request)
                still_waiting.extend(self.waiting)
                break

            self.kv_cache.allocate_for_tokens(request.request_id, request.num_computed_tokens, chunk_size)
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            output.scheduled.append(ScheduledItem(request, num_tokens=chunk_size, is_prefill=True))
            budget -= chunk_size

        self.waiting = still_waiting
        return budget

    # -- cache-space bookkeeping ----------------------------------------------

    def _ensure_cache_space(self, request: Request, num_new_tokens: int,
                             output: Optional[SchedulerOutput] = None, allow_preempt: bool = True) -> bool:
        """Make sure `request` can grow by num_new_tokens, preempting other running
        requests first if that's what it takes (and if we're allowed to)."""
        existing = request.num_computed_tokens
        if self.kv_cache.can_append_tokens(request.request_id, existing, num_new_tokens):
            return True

        if not allow_preempt:
            return False

        # Evict the most-recently-admitted running request first - it's the
        # cheapest to redo later and keeps older requests making progress.
        while self.running and not self.kv_cache.can_append_tokens(request.request_id, existing, num_new_tokens):
            victim = self.running[-1]
            if victim is request:
                break
            self._preempt(victim, output)
            self.running.remove(victim)

        return self.kv_cache.can_append_tokens(request.request_id, existing, num_new_tokens)

    def _preempt(self, request: Request, output: Optional[SchedulerOutput]) -> None:
        """
        Kick a request out of the cache. It keeps everything it has already
        generated but has to redo its KV cache from scratch once it's
        re-admitted - this is the simple "recompute" style of preemption
        (the alternative, swapping to CPU memory, needs real hardware to
        matter and isn't worth modeling here).
        """
        self.kv_cache.free(request.request_id)
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(request)  # jump the line so it re-runs soon
        if output is not None:
            output.preempted.append(request)
