# Continuous Batching Engine

A from-scratch, single-process model of the core mechanics behind
modern LLM serving engines (vLLM, SGLang, TensorRT-LLM): **paged KV
cache allocation**, a **continuous batching scheduler**, and
**chunked prefill**. 

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
