"""
Demo / entry point.

Spins up a tiny LLMEngine with a deliberately small KV cache so you
can actually *see* chunked prefill and preemption happen in the
printed step log, instead of just trusting the code works.

Run:
    python main.py
"""

import random
from typing import List

from config import EngineConfig
from engine.engine import LLMEngine
from engine.model_runner import MockModelRunner


def make_prompt(length: int) -> List[int]:
    """Fake tokenization - just a list of random 'token ids' of a given length."""
    rng = random.Random(length)  # seeded by length, purely so demo output is reproducible
    return [rng.randint(1, 50_000) for _ in range(length)]


def main() -> None:
    config = EngineConfig(
        block_size=4,
        num_gpu_blocks=16,          # only 64 tokens' worth of cache total - deliberately tight
        max_num_batched_tokens=16,   # forces long prompts to be chunked across steps
        max_num_seqs=8,
        max_new_tokens=10,
        eos_token_id=-1,
    )

    model_runner = MockModelRunner(eos_token_id=config.eos_token_id, eos_probability=0.1)
    engine = LLMEngine(config, model_runner)

    # A mix of short and long prompts arriving all at once - long enough
    # that at least one has to be chunked, and enough concurrent requests
    # that cache pressure forces a preemption somewhere in the run.
    prompt_lengths = [6, 20, 3, 14, 8, 25]
    requests = [engine.add_request(make_prompt(n)) for n in prompt_lengths]

    print(f"Engine config: {config}\n")
    print(f"Submitted {len(requests)} requests with prompt lengths {prompt_lengths}\n")

    reports = engine.run_until_complete()
    for report in reports:
        print(report)

    print(f"\nFinished after {len(reports)} steps.\n")
    for request in requests:
        print(f"  req{request.request_id}: prompt_len={request.prompt_len}, "
              f"generated={request.output_len} tokens, status={request.status.name}")


if __name__ == "__main__":
    main()
