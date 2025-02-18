#!/usr/bin/env python3

import ctypes
import binascii

# Define constants
RING_QUEUE_CAPACITY = 128
RING_QUEUE_MASK = RING_QUEUE_CAPACITY - 1

# Define the ring_entry_type enum
class RingEntryType(ctypes.c_int):
    RING_ENTRY_KEY_LEFT  = 1<<0
    RING_ENTRY_KEY_RIGHT = 1<<1
    RING_ENTRY_KEY_UP    = 1<<2
    RING_ENTRY_KEY_DOWN  = 1<<3
    RING_ENTRY_KEY_X     = 1<<4
    RING_ENTRY_KEY_Y     = 1<<5
    RING_ENTRY_KEY_A     = 1<<6
    RING_ENTRY_KEY_B     = 1<<7
    RING_ENTRY_KEY_SAVE  = 1<<8
    RING_ENTRY_KEY_LOAD  = 1<<9
    RING_ENTRY_KEY_PAUSE = 1<<10

    RING_ENTRY_F_SINGLE_TICK_PRESS = 1<<31

# Define the struct ring_entry
class RingEntry(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("data", ctypes.c_uint),
    ]

# Define the struct ring_queue
class RingQueue(ctypes.Structure):
    _fields_ = [
        ("write_idx", ctypes.c_uint),
        ("read_idx", ctypes.c_uint),
        ("array", RingEntry * RING_QUEUE_CAPACITY),
    ]
    def nr_submitted_entries(self):
        """Get number of already submitted entries in the queue."""
        return self.write_idx - self.read_idx

    def get_entry_to_submit(self):
        """Get the next entry to submit, or None if the queue is full."""
        if self.write_idx - self.read_idx >= RING_QUEUE_CAPACITY:
            return None
        return self.array[self.write_idx & RING_QUEUE_MASK]

    def submit(self):
        """Submit the current entry."""
        # TODO: for architectures other than x86-64 we need
        # __atomic_store_n(&write_idx, write_idx + 1, __ATOMIC_RELEASE)
        self.write_idx += 1

    def get_entry_to_retrieve(self):
        """Get the next entry to retrieve, or None if the queue is empty."""
        # TODO: for architectures other than x86-64 we need
	# write_idx = __atomic_load_n(&write_idx, __ATOMIC_ACQUIRE);
        write_idx = self.write_idx
        if write_idx == self.read_idx:
            return None
        return self.array[self.read_idx & RING_QUEUE_MASK]

    def retrieve(self):
        """Mark the current entry as retrieved."""
        self.read_idx += 1

# Example Usage
if __name__ == "__main__":
    import mmap

    # Create shared memory for the ring queue
    shm = mmap.mmap(-1, ctypes.sizeof(RingQueue))

    # Initialize PyRingQueue with the shared memory address
    ring = PyRingQueue(ctypes.addressof(ctypes.c_char.from_buffer(shm)))

    # Initialize the ring queue
    ring.init()

    entry = ring.get_entry_to_retrieve()
    assert(entry is None)

    # Get an entry to submit and submit it
    entry = ring.get_entry_to_submit()
    assert(entry is not None)
    entry.type = 0x666
    entry.data = 0x555
    ring.submit()

    # Retrieve the entry after submitting
    entry = ring.get_entry_to_retrieve()
    assert(entry is not None)
    assert(entry.type == 0x666)
    assert(entry.data == 0x555)

    # Retrieve the next entry, it should be the same as the previous one
    entry2 = ring.get_entry_to_retrieve()
    assert(ctypes.addressof(entry) == ctypes.addressof(entry2))

    # Now retrieve the entry from the ring queue
    ring.retrieve()

    # The ring queue should be empty now
    entry = ring.get_entry_to_retrieve()
    assert(entry is None)
