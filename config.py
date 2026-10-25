"""
Engine-wide configuration knobs.

Keeping these in one place means every other module (cache, scheduler,
engine) reads the *same* numbers instead of hardcoding magic constants
scattered across files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    # How many tokens fit in a single KV cache block (a "page").
    # Real engines use 16 or 32; we use a small number so the demo
    # can actually run out of cache and show preemption happening.
    block_size: int = 4

    # Total number of blocks the "GPU" has. This is our whole memory budget.
    num_gpu_blocks: int = 16

    # Max tokens we're willing to push through the model in ONE step.
    # This is what forces long prompts to be chunked across several
    # steps instead of being prefilled in one giant, decode-blocking shot.
    max_num_batched_tokens: int = 16

    # Max number of requests allowed to run (occupy cache) concurrently.
    max_num_seqs: int = 8

    # Every sequence is cut off here even if it never predicts EOS,
    # so the demo always terminates.
    max_new_tokens: int = 10

    # Token id treated as "end of sequence" by the mock model runner.
    eos_token_id: int = -1
