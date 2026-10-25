"""
A Request is one user prompt moving through the engine's lifecycle:

    WAITING -> RUNNING -> (PREEMPTED -> RUNNING)* -> FINISHED

It owns nothing about *how* it gets scheduled or *where* its KV cache
physically lives - that's the Scheduler's and PagedKVCache's job. This
class is just the piece of state that travels between them.
"""

from enum import Enum, auto
from itertools import count
from typing import List


class RequestStatus(Enum):
    WAITING = auto()     # sitting in the queue, no cache allocated yet
    RUNNING = auto()      # currently admitted - either mid-prefill or decoding
    PREEMPTED = auto()    # was running, got kicked out to free cache space
    FINISHED = auto()      # done - hit EOS or max_new_tokens


class Request:
    """A single prompt plus everything generated for it so far."""

    _id_counter = count()

    def __init__(self, prompt_token_ids: List[int], max_new_tokens: int, arrival_step: int = 0):
        self.request_id: int = next(Request._id_counter)
        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.max_new_tokens = max_new_tokens
        self.arrival_step = arrival_step

        self.status: RequestStatus = RequestStatus.WAITING

        # How many tokens of THIS request currently have a KV entry stored
        # in the cache. This is the single source of truth for "how much
        # cache am I using" and also what makes chunked prefill possible -
        # we don't have to process the whole prompt in one shot, we just
        # need to remember where we left off.
        self.num_computed_tokens: int = 0

        # Block ids this request owns in the PagedKVCache. The cache fills
        # this in; the request just carries the reference around.
        self.block_table: List[int] = []

    # -- convenience helpers -------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_prefill_done(self) -> bool:
        """True once the whole prompt has been run through the model at least once."""
        return self.num_computed_tokens >= self.prompt_len

    @property
    def num_pending_prompt_tokens(self) -> int:
        """How many prompt tokens are still left to chew through (chunked prefill)."""
        return max(0, self.prompt_len - self.num_computed_tokens)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def is_finished(self, eos_token_id: int) -> bool:
        if self.output_token_ids and self.output_token_ids[-1] == eos_token_id:
            return True
        return self.output_len >= self.max_new_tokens

    def __repr__(self) -> str:
        return (f"Request(id={self.request_id}, status={self.status.name}, "
                f"prompt_len={self.prompt_len}, output_len={self.output_len})")
