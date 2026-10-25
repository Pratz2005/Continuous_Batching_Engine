# Continuous Batching Engine (minimal, pure Python)

A from-scratch, single-process model of the core mechanics behind
modern LLM serving engines (vLLM, SGLang, TensorRT-LLM): **paged KV
cache allocation**, a **continuous batching scheduler**, and
**chunked prefill**. No GPU, no model weights, no external
dependencies — just the standard library, so the scheduling and
memory logic is the whole point and nothing is hidden behind a
framework.

Since there's no real model here, a `MockModelRunner` stands in for
the forward pass (see "Simplifications" below). Everything about
*scheduling what runs when* and *how KV cache memory is managed* is
real.

## Run it

```bash
python main.py
```

This prints one line per scheduler step, e.g.:

```
step   3 | batch: req1:decode, req3:prefill+9, req0:decode
```

so you can watch decode and prefill get interleaved in the same step,
chunks of a long prompt spread across multiple steps, and (with the
default tight cache config) a preemption get triggered.

Run the tests:

```bash
python -m unittest discover -v
```

## Project layout

```
config.py                  # EngineConfig - every tunable knob in one place
engine/
  request.py                # Request + RequestStatus - the state machine for one prompt
  kv_cache.py                # Block, BlockAllocator, PagedKVCache - the paging allocator
  chunked_prefill.py         # PrefillChunker - how much of a prompt to run this step
  scheduler.py                # ContinuousBatchingScheduler - the actual scheduling policy
  model_runner.py             # BaseModelRunner / MockModelRunner - stands in for the model
  engine.py                    # LLMEngine - wires scheduler + cache + model runner together
main.py                      # demo script
tests/test_engine.py          # unit tests for allocator, chunker, and scheduler behavior
```

## How the three concepts map to code

**Paged KV cache** (`engine/kv_cache.py`) — `BlockAllocator` hands out
fixed-size blocks from a free list, like pages of virtual memory.
`PagedKVCache` maps each request to a `block_table` (its "page
table") and only grows it block-by-block as the request needs more
space. No request ever reserves memory for its hypothetical maximum
length up front, so there's no internal fragmentation.

**Continuous batching scheduler** (`engine/scheduler.py`) — on every
`schedule()` call it rebuilds the batch from scratch: requests that
have already finished prefill get a cheap 1-token decode step first,
then in-progress prefills get continued with whatever token budget is
left, then brand-new requests get admitted if there's still room.
Requests join and leave every single iteration — nobody waits for an
entire batch to finish before a new request can start.

**Chunked prefill** (`engine/chunked_prefill.py`) — a shared
`max_num_batched_tokens` budget per step means a long prompt can't
monopolize a step. `PrefillChunker` just caps a prefill request to
whatever's left of the budget; the request's `num_computed_tokens`
tracks how far into its own prompt it's gotten, so the next step picks
up exactly where it left off.

**Bonus: preemption** — if the cache is full and a running request
needs more blocks, the scheduler evicts the most-recently-admitted
running request (`_preempt` in `scheduler.py`), freeing its blocks.
The victim keeps its already-generated tokens but recomputes its KV
cache from scratch once it's re-admitted ("recompute" preemption —
the simpler alternative to swapping to CPU memory).

## Simplifications (deliberate, worth being able to explain)

- **No real model.** `MockModelRunner` fakes the forward pass. The
  resume bullet is about memory/scheduling mechanics, not about
  training or loading weights, so a real model would add dependency
  weight without adding to what the project demonstrates.
- **Single process, not actually distributed.** "Distributed" serving
  in production is about routing requests across multiple
  model-replica workers and sharding KV cache across GPUs/nodes. This
  project models the per-worker scheduling loop that *each* node in a
  distributed setup would run — the natural next step would be adding
  a router process that fans requests out to several `LLMEngine`
  instances.
- **No copy-on-write block sharing.** Real engines share prompt blocks
  across requests doing beam search or prefix caching (ref-counted
  blocks). `Block.ref_count` exists but is always 1 here — there's
  only ever one owner per block.
- **Prefill's last chunk doesn't fuse with the first decode token.**
  Real engines produce the first output token as part of the same
  forward pass that finishes the prompt. Here that's one extra,
  separate step for simplicity — cheap to fix, not worth the
  complexity for what this demonstrates.

## Talking points if asked about this in an interview

- *"Why paged blocks instead of one contiguous buffer per request?"*
  Contiguous buffers sized for the worst case waste memory on every
  request that doesn't hit its max length, and freeing/reallocating
  variable-size buffers fragments memory over time. Fixed-size blocks
  from a shared free list sidestep both problems — same idea as OS
  virtual memory paging.
- *"Why prioritize decode over prefill in the scheduler?"* Decode
  steps are cheap (1 token) and serve users who are already mid-stream
  getting output; starving them to prefill a new request creates
  visible per-token latency spikes. Chunking prefill instead of
  blocking on it is what keeps decode latency smooth.
- *"What's the actual cost of preemption?"* Whatever work was spent on
  the victim's KV cache is thrown away and redone from scratch next
  time it's scheduled — so preemption trades memory pressure for
  wasted compute. That's exactly the kind of trade-off a real
  scheduler's admission policy has to be tuned around.
