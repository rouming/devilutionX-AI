#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

#include "items.h"
#include "monster.h"
#include "levels/gendung.h"
#include "utils/ring.h"
#include "utils/shared.h"

namespace devilution {

#define ATTRS(align)								\
	__attribute__((section(".shared"),no_reorder))	\
	__attribute__ ((aligned(align)))

struct entry_ptr {
	const char       *name;
	void             *addr;
	size_t           align;
	size_t           size;
	struct entry_ptr *next;
};

static struct entry_ptr *registered_ptr;

#define DEFINE_VAR(aligned, type, named)		\
	ATTRS(aligned) __typeof__(type) named;		\
	__attribute__((constructor))				\
	static void register_##named(void)			\
	{											\
		static struct entry_ptr ptr = {			\
			.name = #named,						\
			.addr = &named,						\
			.align = aligned,					\
			.size = sizeof(named),				\
			.next = registered_ptr,				\
		};										\
		registered_ptr = &ptr;					\
	}											\

/*
 * Here we define all variables which should be shared
 */
namespace shared {
	DEFINE_VAR(2, uint16_t[2], maxdun);
	DEFINE_VAR(2, uint16_t[2], dmax);
	DEFINE_VAR(2, uint16_t, max_monsters);
	DEFINE_VAR(2, uint16_t, max_objects);
	DEFINE_VAR(2, uint16_t, max_tiles);
	DEFINE_VAR(2, uint16_t, num_mtypes);
	DEFINE_VAR(4, struct ring_queue, input_queue);
	DEFINE_VAR(4, struct ring_queue, events_queue);
	DEFINE_VAR(4, struct player_state, player);
	DEFINE_VAR(8, uint64_t, game_tick);
}
/* monsters.cpp */
DEFINE_VAR(8, size_t, LevelMonsterTypeCount);
DEFINE_VAR(8, size_t, ActiveMonsterCount);
DEFINE_VAR(4, Monster[MaxMonsters], Monsters);
DEFINE_VAR(4, unsigned[MaxMonsters], ActiveMonsters);
DEFINE_VAR(4, int[NUM_MTYPES], MonsterKillCounts);

/* objects.cpp */
DEFINE_VAR(4, Object[MAXOBJECTS], Objects);

/* items.cpp, gendung.cpp, automap.cpp */
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dItem);
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dTransVal);
DEFINE_VAR(4, DungeonFlag[MAXDUNX][MAXDUNY], dFlags);
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dPlayer);
DEFINE_VAR(4, int16_t[MAXDUNX][MAXDUNY], dMonster);
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dCorpse);
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dObject);
DEFINE_VAR(4, uint16_t[MAXDUNX][MAXDUNY], dPiece);
DEFINE_VAR(4, int8_t[MAXDUNX][MAXDUNY], dSpecial);
DEFINE_VAR(4, uint8_t[DMAXX][DMAXY], AutomapView);
DEFINE_VAR(4, TileProperties[MAXTILES], SOLData);


/*
 * End of sharing
 */

/* See the script.ld for the details */
extern "C" {
	int shared_section_start;
	int shared_section_end;
}

static void verify_no_padding(void)
{
	struct entry_ptr *ptr, *prev;

	/* Reverse chain */
	prev = NULL;
	ptr = registered_ptr;
	do {
		struct entry_ptr *next;
		next = ptr->next;
		ptr->next = prev;
		prev = ptr;
		ptr = next;
	} while (ptr);
	registered_ptr = prev;

	/* Verify no padding */
	prev = NULL;
	for (ptr = registered_ptr; ptr; ptr = ptr->next) {
		if (!prev) {
			assert((char *)ptr->addr == (char *)&shared_section_start);
		} else {
			if ((char *)prev->addr + prev->size != (char *)ptr->addr) {
				printf("!!! %s (%p + %ld) != %s (%p)\n",
					   prev->name, prev->addr, prev->size,
					   ptr->name, ptr->addr);
				assert(false);
			}
		}
		prev = ptr;
	}
	assert(prev->addr <= &shared_section_end);
}

static void do_init_after_map(void)
{
	using namespace shared;

	maxdun[0] = MAXDUNX;
	maxdun[1] = MAXDUNY;
	dmax[0] = DMAXX;
	dmax[1] = DMAXY;
	max_monsters = MaxMonsters;
	max_objects  = MAXOBJECTS;
	max_tiles    = MAXTILES;
	num_mtypes   = NUM_MTYPES;

	ring_queue_init(&input_queue);
	ring_queue_init(&events_queue);
}

__attribute__((constructor))
static void do_map(void)
{
	int fd, ret;
	void *addr;
	size_t len;

	verify_no_padding();

#if 0
	using namespace shared;

	printf(">> sizeof(struct object)=%ld, %ld, %ld, %ld, %ld\n",
		   sizeof(struct Object),
		   (size_t)&((struct Object *)0)->position,
		   (size_t)&((struct Object *)0)->_oMissFlag,
		   (size_t)&((struct Object *)0)->selectionRegion,
		   (size_t)&((struct Object *)0)->_oPreFlag


		);

	printf(">> Player size=%ld, _pNumInv=%ld, _pmode=%ld\n",
		   sizeof(player),
		   (size_t)&((struct player_state *)0)->_pNumInv,
		   (size_t)&((struct player_state *)0)->_pmode);

	printf(">> Monster size=%ld\n",
		   sizeof(Monster));

	exit(0);
#endif

	len = (char *)&shared_section_end - (char *)&shared_section_start;

	fd = open("/tmp/diablo.shared", O_CREAT | O_TRUNC | O_RDWR,
				  S_IRUSR | S_IWUSR);
	if (fd < 0) {
		perror("open: ");
		exit(1);
	}
	ret = ftruncate(fd, len);
	if (ret < 0) {
		perror("ftruncate: ");
		exit(1);
	}
	addr = mmap(&shared_section_start, len,
				PROT_READ | PROT_WRITE,
				MAP_SHARED | MAP_FIXED,
				fd, 0);
	close(fd);
	if (addr == MAP_FAILED) {
		perror("mmap: ");
		exit(1);
	}

	do_init_after_map();
}

}