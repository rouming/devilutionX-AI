#ifndef RING_H
#define RING_H

#include <stddef.h>

#define RING_QUEUE_CAPACITY 128
#define RING_QUEUE_MASK     (RING_QUEUE_CAPACITY - 1)

enum ring_entry_type {
	RING_ENTRY_KEY_LEFT  = 1<<0,
	RING_ENTRY_KEY_RIGHT = 1<<1,
	RING_ENTRY_KEY_UP    = 1<<2,
	RING_ENTRY_KEY_DOWN  = 1<<3,
	RING_ENTRY_KEY_X     = 1<<4,
	RING_ENTRY_KEY_Y     = 1<<5,
	RING_ENTRY_KEY_A     = 1<<6,
	RING_ENTRY_KEY_B     = 1<<7,
	RING_ENTRY_KEY_SAVE  = 1<<8,
	RING_ENTRY_KEY_LOAD  = 1<<9,
	RING_ENTRY_KEY_PAUSE = 1<<10,

	RING_ENTRY_F_SINGLE_TICK_PRESS = 1<<31,

	RING_ENTRY_FLAGS     = (RING_ENTRY_F_SINGLE_TICK_PRESS)
};

struct ring_entry {
	uint32_t type;
	uint32_t data;
};

struct ring_queue {
	uint32_t          write_idx;
	uint32_t          read_idx;
	struct ring_entry array[RING_QUEUE_CAPACITY];
};

static inline void ring_queue_init(struct ring_queue *ring)
{
	*ring = (struct ring_queue){};
}

static inline struct ring_entry *
ring_queue_get_entry_to_submit(struct ring_queue *ring)
{
	if (ring->write_idx - ring->read_idx >= RING_QUEUE_CAPACITY)
		return NULL;

	return &ring->array[ring->write_idx & RING_QUEUE_MASK];
}

static inline void ring_queue_submit(struct ring_queue *ring)
{
	__atomic_store_n(&ring->write_idx, ring->write_idx + 1,
					 __ATOMIC_RELEASE);
}

static inline struct ring_entry *
ring_queue_get_entry_to_retreive(struct ring_queue *ring)
{
	unsigned write_idx;

	write_idx = __atomic_load_n(&ring->write_idx, __ATOMIC_ACQUIRE);
	if (write_idx == ring->read_idx)
		return NULL;

	return &ring->array[ring->read_idx & RING_QUEUE_MASK];
}

static inline void ring_queue_retrieve(struct ring_queue *ring)
{
	ring->read_idx++;
}


/* #define TEST */
#ifdef  TEST

#include <assert.h>

int main()
{
	struct ring_queue ring;
	struct ring_entry *entry, *entry2;

	ring_queue_init(&ring);

	entry = ring_queue_get_entry_to_retreive(&ring);
	assert(entry == NULL);

	entry = ring_queue_get_entry_to_submit(&ring);
	assert(entry != NULL);
	entry->type = 666;

	ring_queue_submit(&ring);
	entry = ring_queue_get_entry_to_retreive(&ring);
	assert(entry != NULL);
	assert(entry->type == 666);

	entry2 = ring_queue_get_entry_to_retreive(&ring);
	assert(entry == entry2);

	ring_queue_retrieve(&ring);
	entry = ring_queue_get_entry_to_retreive(&ring);
	assert(entry == NULL);
}

#endif /* TEST */

#endif /* RING_H */
