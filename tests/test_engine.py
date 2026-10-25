"""
Sanity tests for the three load-bearing mechanics: block allocation,
chunked prefill splitting, and preemption under cache pressure.

Run from the project root with:
    python -m unittest discover -v
"""

import unittest

from config import EngineConfig
from engine.chunked_prefill import PrefillChunker
from engine.engine import LLMEngine
from engine.kv_cache import BlockAllocator, PagedKVCache
from engine.model_runner import MockModelRunner
from engine.request import Request, RequestStatus


class TestBlockAllocator(unittest.TestCase):
    def test_allocate_until_exhausted_then_raises(self):
        allocator = BlockAllocator(num_blocks=2)
        allocator.allocate()
        allocator.allocate()
        with self.assertRaises(MemoryError):
            allocator.allocate()

    def test_free_returns_block_to_pool(self):
        allocator = BlockAllocator(num_blocks=1)
        block = allocator.allocate()
        self.assertEqual(allocator.num_free_blocks, 0)
        allocator.free(block.block_id)
        self.assertEqual(allocator.num_free_blocks, 1)


class TestPagedKVCache(unittest.TestCase):
    def test_block_math_is_ceiling_division(self):
        cache = PagedKVCache(num_blocks=4, block_size=4)
        # 5 tokens need 2 blocks of size 4 (4 + 1), not 1.
        self.assertTrue(cache.can_append_tokens(request_id=1, num_existing_tokens=0, num_new_tokens=5))
        cache.allocate_for_tokens(request_id=1, num_existing_tokens=0, num_new_tokens=5)
        self.assertEqual(len(cache.get_block_table(1)), 2)

    def test_free_releases_blocks_for_reuse(self):
        cache = PagedKVCache(num_blocks=2, block_size=4)
        cache.allocate_for_tokens(request_id=1, num_existing_tokens=0, num_new_tokens=8)
        self.assertEqual(cache.num_free_blocks(), 0)
        cache.free(1)
        self.assertEqual(cache.num_free_blocks(), 2)


class TestPrefillChunker(unittest.TestCase):
    def test_chunk_is_capped_by_budget(self):
        chunker = PrefillChunker(max_num_batched_tokens=16)
        self.assertEqual(chunker.next_chunk_size(num_pending_prompt_tokens=20, tokens_left_in_budget=16), 16)

    def test_chunk_is_capped_by_remaining_prompt(self):
        chunker = PrefillChunker(max_num_batched_tokens=16)
        self.assertEqual(chunker.next_chunk_size(num_pending_prompt_tokens=3, tokens_left_in_budget=16), 3)


class TestRequest(unittest.TestCase):
    def test_finishes_on_eos(self):
        request = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=10)
        request.append_output_token(-1)
        self.assertTrue(request.is_finished(eos_token_id=-1))

    def test_finishes_on_max_new_tokens(self):
        request = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=2)
        request.append_output_token(99)
        request.append_output_token(100)
        self.assertTrue(request.is_finished(eos_token_id=-1))


class TestSchedulerBehavior(unittest.TestCase):
    """These exercise the engine end-to-end, which is really exercising the scheduler."""

    def _make_engine(self, **overrides) -> LLMEngine:
        config = EngineConfig(**overrides)
        runner = MockModelRunner(eos_token_id=config.eos_token_id, eos_probability=0.0)
        return LLMEngine(config, runner)

    def test_long_prompt_gets_chunked_across_multiple_steps(self):
        # 20-token prompt, but only 8 tokens of budget per step -> must take
        # more than one step just to finish the prefill.
        engine = self._make_engine(block_size=4, num_gpu_blocks=16, max_num_batched_tokens=8, max_new_tokens=1)
        request = engine.add_request(prompt_token_ids=list(range(20)))

        engine.step()
        self.assertFalse(request.is_prefill_done)
        self.assertEqual(request.num_computed_tokens, 8)

        engine.step()
        self.assertFalse(request.is_prefill_done)
        self.assertEqual(request.num_computed_tokens, 16)

    def test_cache_pressure_triggers_preemption(self):
        # Exactly 2 blocks of size 4 = 8 tokens of cache, total - just enough
        # for both 4-token prompts to be admitted, but not enough for BOTH
        # to grow by a token during decode at the same time. The second one
        # admitted should get evicted to make room for the first.
        engine = self._make_engine(block_size=4, num_gpu_blocks=2, max_num_batched_tokens=16,
                                     max_num_seqs=8, max_new_tokens=3)
        engine.add_request(prompt_token_ids=list(range(4)))
        engine.add_request(prompt_token_ids=list(range(4)))

        preempted_ever = False
        for _ in range(10):
            if not engine.has_unfinished_requests:
                break
            report = engine.step()
            if report.output.preempted:
                preempted_ever = True

        self.assertTrue(preempted_ever, "expected cache pressure to force at least one preemption")

    def test_waiting_request_is_not_lost_behind_one_that_does_not_fit(self):
        # Regression test: if the request at the FRONT of the waiting queue
        # can't fit this step, everyone still behind it must stay in the
        # queue too - not get silently dropped when admission bails out.
        engine = self._make_engine(block_size=4, num_gpu_blocks=16, max_num_batched_tokens=16,
                                     max_num_seqs=8, max_new_tokens=1)
        # First request eats the whole token budget for this step...
        big = engine.add_request(prompt_token_ids=list(range(16)))
        # ...so this one can't be admitted yet, but must survive to the next step.
        small = engine.add_request(prompt_token_ids=list(range(2)))

        engine.step()
        self.assertIn(small, engine.scheduler.waiting)

        engine.run_until_complete()
        self.assertEqual(big.status, RequestStatus.FINISHED)
        self.assertEqual(small.status, RequestStatus.FINISHED)

    def test_finished_requests_free_their_cache(self):
        engine = self._make_engine(block_size=4, num_gpu_blocks=16, max_num_batched_tokens=16, max_new_tokens=1)
        engine.add_request(prompt_token_ids=[1, 2, 3])

        engine.run_until_complete()
        self.assertEqual(engine.kv_cache.num_free_blocks(), 16)


if __name__ == "__main__":
    unittest.main()
