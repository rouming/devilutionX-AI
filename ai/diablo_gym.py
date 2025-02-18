"""
diablo_gym.py - Diablo Gymnasium environment
"""
import enum
import numpy as np
import gymnasium as gym

import diablo_state

# Return neighbours in cardinal CW directions:
# N, NE, E, SE, S, SW, W, NW
def get_matrix_neighbors(matrix, i, j):
    rows, cols = matrix.shape
    neighbors = np.full(8, -1, dtype=np.int32)
    ind = 0
    for y in range(j-1, j+2):
        for x in range(i-1, i+2):
            # Exclude center element
            if (x, y) == (i, j):
                continue
            ind += 1
            if x < 0 or x >= rows or y < 0 or y >= cols:
                continue
            neighbors[ind-1] = matrix[y, x]

    # Remap array from:
    # "NW, N, NE, W, E, SW, S, SE" to "N, NE, E, SE, S, SW, W, NW"
    # to have a simple mapping to ActionEnum
    remap =[1, 2, 4, 7, 6, 5, 3, 0]
    return np.take(neighbors, remap)

class DiabloEnv(gym.Env):
    class ActionEnum(enum.Enum):
        Walk_N          = enum.auto()
        Walk_NE         = enum.auto()
        Walk_E          = enum.auto()
        Walk_SE         = enum.auto()
        Walk_S          = enum.auto()
        Walk_SW         = enum.auto()
        Walk_W          = enum.auto()
        Walk_NW         = enum.auto()
        Stand           = enum.auto()
        # Attack monsters, talk to towners, lift and place inventory items.
        PrimaryAction   = enum.auto()
        # Open chests, interact with doors, pick up items.
        SecondaryAction = enum.auto()

    @staticmethod
    def action_to_key(action):
        match ActionEnum(action):
            case ActionEnum.Walk_N:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_NE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_E:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_SE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
            case ActionEnum.Walk_S:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_SW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_W:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_NW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP)
            case ActionEnum.Stand:
                # No key - NOP
                key = 0
            case ActionEnum.PrimaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_A)
            case ActionEnum.SecondaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_B)
        return key

    def submit_key_and_wait_for_next_tick(self, key):
        entry = self.game.state.input_queue.get_entry_to_submit()
        assert entry
        entry.type = key
        entry.data = 0

        # Submit key
        self.game.state.input_queue.submit()

        # Busy-loop for actual key acceptance
        while self.game.state.input_queue.nr_entries_to_submit() != \
              ring.RING_QUEUE_CAPACITY:
            time.sleep(0.01)

        # Busy-loop for next tick after key was injected
        tick = self.game.state.game_tick
        while self.game.state.game_tick == tick:
            time.sleep(0.01)

    def __init__(self, env_config):
        self.config = env_config
        self.seed = env_config["seed"] + env_config.worker_index

        self.game = diablo_state.DiabloGame.run(env_config)

        # Submit SAVE, so we LOAD on reset()
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.submit_key_and_wait_for_next_tick(key)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = self.diablo_state.get_environment(d)

        self.action_space = gym.spaces.Discrete(len(ActionEnum))

        dtype = np.uint16
        high = (1 << len(game_state.EnvironmentFlag)) - 1
        # Don't forget to change dtype if does not fit max number of bits
        assert high <= np.iinfo(dtype).max
        self.observation_space = gym.spaces.Dict(
            {
                "env-status":  gym.spaces.Discrete(env_status.shape[0]),
                "environment": gym.spaces.Box(low=0, high=high,
                                              shape=env.shape,
                                              dtype=dtype),
            }
        )

    def get_env_status(self, d):
        kills = np.sum(d.MonsterKillCounts_np)
        hp = d.player._pHitPoints
        mode = d.player._pmode
        return np.array([kills, hp, mode])

    def action_mask(self, env):
        """Computes an action mask for the action space using the state information."""
        mask = np.full(len(ActionEnum), 1, dtype=np.int8)
        player_pos = np.array([d.player.position.future.x,
                               d.player.position.future.y])

        neighbors = get_matrix_neighbors(d, *player_pos)
        for i, tile in enumerate(neighbors):
            if tile < 0:
                # Block in the direction beyond map
                mask[i] = 0
                continue

            if tile & game_state.EnvironmentFlag.Trigger.value:
                # Block in the direction of any trigger, so forbid
                # escaping to the town
                # TODO: this blocks all the trigger including stairs to the
                # next level
                mask[i] = 0
            elif tile & game_state.EnvironmentFlag.Wall.value or \
                 tile & game_state.EnvironmentFlag.DoorClosed.value or \
                 tile & game_state.EnvironmentFlag.Barrel.value or \
                 tile & game_state.EnvironmentFlag.ChestClosed.value or \
                 tile & game_state.EnvironmentFlag.ChestOpened.value or \
                 tile & game_state.EnvironmentFlag.SarcophClosed.value or \
                 tile & game_state.EnvironmentFlag.SarcophOpened.value:
                # Block move through solid objects
                mask[i] = 0

        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)

        # Submit LOAD
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.submit_key_and_wait_for_next_tick(key)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = self.diablo_state.get_environment(d)

        obj_cnt = diablo_state.count_active_objects(d)
        items_cnt = diablo_state.count_active_items(d)
        monsers_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_sate.count_explored_tiles(d)
        hp = d.player._pHitPoints

        self.prev_obj_cnt = obj_cnt
        self.prev_items_cnt = items_cnt
        self.prev_monsters_cnt = monsters_cnt
        self.prev_total_hp = total_hp
        self.prev_explored_cnt = explored_cnt
        self.prev_hp = hp

        obs = {
            "env-status":  env_status,
            "environment": env,
        }

        # Probability is always 1.0, diablo environment is deterministic
        info = {"prob": 1.0, "action_mask": self.action_mask(env)}

        return obs, info

    def evalulate_step(self, d, env, action):
        # truncated flag indicates whether the episode was forcefully
        # stopped due to a time limit or other constraints not related
        # to task completion.
        truncated = False
        done = False

        obj_cnt = diablo_state.count_active_objects(d)
        items_cnt = diablo_state.count_active_items(d)
        monsers_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_sate.count_explored_tiles(d)
        hp = d.player._pHitPoints

        # Exploration reward
        # +1. calculate number of explored tiles, compare with a previous step
        # 2. find a way to estimate full exploration
        # 3. penalty for revisiting explored tiles (needed?)

        # Combat reward
        # +1. calculate number of all monsters HP, compare with previous step,
        #    reward for reduced, i.e. monster damage
        # +2.   reward for killing monster
        # 2.1. huge reward once last monster is killed

        # Survival & Caution Reward
        # +1.  penalty for taking damage
        # +1.1 higher penalty when HP drops beyond threshold
        # 2.  reward for dodging an attack successfully
        # 3.  reward for staying at high health

        # Strategic Encouragement
        # +1. reward for collecting items
        # 2. penalty for attacking walls or wasting actions
        # 3. reward for reaching the next level stairs after full explorations

        # Endgame conditions
        # +1. Huge penalty for dying
        # 2. Huge penalty for escaping to the town
        # 3. Huge penalty for descending if current dungeon is not cleared

        if explored_cnt > self.prev_explored_cnt:
            # Exploration
            reward = 1
        elif obj_cnt < self.prev_obj_cnt:
            # Reward for opening doors, chests, etc
            reward = 5
        elif items_cnt < self.prev_obj_cnt:
            # Reward for collecting items. Item number can increase if
            # dropped from a chest, but actions can't be applied on a
            # closed chest and picking up an item, so object counter
            # should be checked first
            reward = 5
        elif total_hp < self.total_hp:
            if monsters_cnt < self.monsters_cnt:
                # Monster killed
                reward = 20
            else:
                # Monster took damage
                reward = 10
        elif hp < self.prev_hp:
            if hp <= 0:
                # We are dead, game over
                reward = -1000
                done = True
            else:
                # More damage to health - more penalty
                # from 0 to -100
                change = (hp - self.prev_hp) / self.prev_hp
                reward = change * 100
        else:
            # Penalty for NOP
            reward = -0.1

        self.prev_obj_cnt = obj_cnt
        self.prev_items_cnt = items_cnt
        self.prev_monsters_cnt = monsters_cnt
        self.prev_total_hp = total_hp
        self.prev_explored_cnt = explored_cnt
        self.prev_hp = hp

        return reward, done, truncated

    def step(self, action):
        key = action_to_key(action)
        key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS

        # Submit
        self.submit_key_and_wait_for_next_tick(key)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = self.diablo_state.get_environment(d)

        reward, done, truncated = self.evalulate_step(d, env, action)

        obs = {
            "env-status":  env_status,
            "environment": env,
        }
        # Probability is always 1.0, diablo environment is deterministic
        info = {"prob": 1.0, "action_mask": self.action_mask(env)}

        return obs, reward, done, truncated, info
