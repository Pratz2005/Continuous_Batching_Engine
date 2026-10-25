"""
LLMEngine wires the three pieces together:
  - ContinuousBatchingScheduler decides *what* runs this step
  - BaseModelRunner actually "runs" it
  - PagedKVCache is what the scheduler consults for memory bookkeeping

This mirrors how real serving engines are structured: the engine
itself stays thin, it just drives the step() loop and applies results
back onto request state.
"""

from typing import List

from config import EngineConfig
from engine.kv_cache import PagedKVCache
from engine.model_runner import BaseModelRunner
from engine.request import Request, RequestStatus
from engine.scheduler import ContinuousBatchingScheduler, SchedulerOutput


class StepReport:
    """Small summary of one engine.step() call - mostly for printing/demo purposes."""

    def __init__(self, step_index: int, output: SchedulerOutput, finished: List[Request]):
        self.step_index = step_index
        self.output = output
        self.finished = finished

    def __str__(self) -> str:
        parts = []
        for item in self.output.scheduled:
            kind = f"prefill+{item.num_tokens}" if item.is_prefill else "decode"
            parts.append(f"req{item.request.request_id}:{kind}")
        batch_desc = ", ".join(parts) if parts else "(idle)"

        preempt_desc = ""
        if self.output.preempted:
            preempt_desc = f" | preempted: {[r.request_id for r in self.output.preempted]}"

        finished_desc = ""
        if self.finished:
            finished_desc = f" | finished: {[r.request_id for r in self.finished]}"

        return f"step {self.step_index:>3} | batch: {batch_desc}{preempt_desc}{finished_desc}"


class LLMEngine:
    """Top-level orchestrator - what a user/caller actually talks to."""

    def __init__(self, config: EngineConfig, model_runner: BaseModelRunner):
        self.config = config
        self.kv_cache = PagedKVCache(config.num_gpu_blocks, config.block_size)
        self.scheduler = ContinuousBatchingScheduler(config, self.kv_cache)
        self.model_runner = model_runner
        self._step_count = 0

    def add_request(self, prompt_token_ids: List[int]) -> Request:
        request = Request(prompt_token_ids, self.config.max_new_tokens, arrival_step=self._step_count)
        self.scheduler.add_request(request)
        return request

    @property
    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def step(self) -> StepReport:
        """Run exactly one iteration: schedule a batch, execute it, update state."""
        output = self.scheduler.schedule()
        next_tokens = self.model_runner.forward(output)

        finished: List[Request] = []
        for item in output.scheduled:
            request = item.request

            if item.is_prefill:
                request.num_computed_tokens += item.num_tokens
                continue  # still digesting the prompt, no output token yet

            token_id = next_tokens[request.request_id]
            request.append_output_token(token_id)

            if request.is_finished(self.config.eos_token_id):
                request.status = RequestStatus.FINISHED
                self.kv_cache.free(request.request_id)
                self.scheduler.running.remove(request)
                finished.append(request)

        self._step_count += 1
        return StepReport(self._step_count, output, finished)

    def run_until_complete(self, max_steps: int = 1000) -> List[StepReport]:
        """Convenience driver for the demo - keeps stepping until nothing's left to do."""
        reports = []
        while self.has_unfinished_requests and self._step_count < max_steps:
            reports.append(self.step())
        return reports
