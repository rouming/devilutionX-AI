#pragma once

#include "utils/ring.h"
#include "player.h"

namespace devilution {

namespace shared {
	extern struct ring_queue   input_queue;
	extern struct ring_queue   events_queue;
	extern uint64_t            game_tick;
	extern struct player_state player;
}

}