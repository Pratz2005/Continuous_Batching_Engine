"""
Paged KV cache.

The whole point of this file is to stop giant contiguous allocations
from fragmenting memory. Instead of giving a request one big slab
sized for its *maximum* possible length up front, we hand out
fixed-size blocks ("pages") on demand - exactly like an OS paging
virtual memory. A request's block_table is basically its page table.
"""

from typing import Dict, List


class Block:
    """One fixed-size chunk of KV cache storage."""

    __slots__ = ("block_id", "ref_count")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0  # how many requests currently point at this block

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, ref_count={self.ref_count})"


class BlockAllocator:
    """Owns the free-list of physical blocks. Just a stack of free ids."""

    def __init__(self, num_blocks: int):
        self._blocks = [Block(i) for i in range(num_blocks)]
        self._free_block_ids: List[int] = list(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def allocate(self) -> Block:
        if not self._free_block_ids:
            raise MemoryError("BlockAllocator: out of KV cache blocks")
        block_id = self._free_block_ids.pop()
        block = self._blocks[block_id]
        block.ref_count = 1
        return block

    def free(self, block_id: int) -> None:
        block = self._blocks[block_id]
        block.ref_count = 0
        self._free_block_ids.append(block_id)


class PagedKVCache:
    """
    Tracks, per request, which physical blocks hold its KV entries.

    This class deliberately knows nothing about *scheduling* - it just
    answers "can I fit N more tokens?" and "give me N more tokens of
    space" / "free this request's space". The scheduler decides when
    to actually call these.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        # request_id -> list of block ids it owns, in order.
        self.block_tables: Dict[int, List[int]] = {}

    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    def _blocks_required(self, num_tokens: int) -> int:
        """Block-aligned ceiling division - how many blocks to hold num_tokens."""
        return -(-num_tokens // self.block_size)

    def can_append_tokens(self, request_id: int, num_existing_tokens: int, num_new_tokens: int) -> bool:
        current_blocks = len(self.block_tables.get(request_id, []))
        total_blocks_needed = self._blocks_required(num_existing_tokens + num_new_tokens)
        extra_blocks_needed = max(0, total_blocks_needed - current_blocks)
        return extra_blocks_needed <= self.num_free_blocks()

    def allocate_for_tokens(self, request_id: int, num_existing_tokens: int, num_new_tokens: int) -> None:
        """Grow (or create) a request's block table to fit num_new_tokens more tokens."""
        table = self.block_tables.setdefault(request_id, [])
        total_blocks_needed = self._blocks_required(num_existing_tokens + num_new_tokens)
        while len(table) < total_blocks_needed:
            block = self.allocator.allocate()
            table.append(block.block_id)

    def free(self, request_id: int) -> None:
        """Release every block a finished/preempted request was holding."""
        table = self.block_tables.pop(request_id, [])
        for block_id in table:
            self.allocator.free(block_id)

    def get_block_table(self, request_id: int) -> List[int]:
        return self.block_tables.get(request_id, [])
