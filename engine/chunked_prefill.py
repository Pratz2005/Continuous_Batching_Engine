"""
Chunked prefill.

Decide how many prompt tokens of a request we're allowed to crunch
through *this step*, given a shared token budget for the whole batch.

Without this, one 2000-token prompt would hog an entire step and every
decode-only request behind it would stall waiting for it to finish -
that's exactly the "long prefill blocks ongoing decodes" problem
continuous batching engines exist to avoid.
"""


class PrefillChunker:
    def __init__(self, max_num_batched_tokens: int):
        self.max_num_batched_tokens = max_num_batched_tokens

    def next_chunk_size(self, num_pending_prompt_tokens: int, tokens_left_in_budget: int) -> int:
        """
        Never ask for more tokens than are left in this step's budget,
        and never more than the prompt actually still needs.
        """
        return max(0, min(num_pending_prompt_tokens, tokens_left_in_budget))
