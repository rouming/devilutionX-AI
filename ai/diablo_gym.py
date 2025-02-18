"""
diablo_gym.py - Diablo Gymnasium environment
"""
import enum
import numpy as np
import gymnasium as gym
import time
import copy

import diablo_state
import ring

class DiabloEnv(gym.Env):
    class KeyWaitFlags(enum.Enum):
        WAIT_KEY_ACCEPTANCE = 1<<0
        WAIT_FOR_NEXT_TICK  = 1<<1

    DONT_WAIT                         = 0
    WAIT_KEY_ACCEPTANCE_ONLY          = (KeyWaitFlags.WAIT_KEY_ACCEPTANCE.value)
    WAIT_KEY_ACCEPTANCE_AND_NEXT_TICK = (KeyWaitFlags.WAIT_KEY_ACCEPTANCE.value |
                                         KeyWaitFlags.WAIT_FOR_NEXT_TICK.value)

    class ActionMask(enum.Enum):
        MASK_TRIGGERS      = 1<<0
        MASK_CLOSED_DOORS  = 1<<1
        MASK_WALLS         = 1<<2
        MASK_OTHER_SOLIDS  = 1<<3

    MASK_EVERYTHING = (ActionMask.MASK_TRIGGERS.value |
                       ActionMask.MASK_CLOSED_DOORS.value |
                       ActionMask.MASK_WALLS.value |
                       ActionMask.MASK_OTHER_SOLIDS.value)

    class ActionEnum(enum.Enum):
        Walk_N          = 0
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
        match DiabloEnv.ActionEnum(action):
            case DiabloEnv.ActionEnum.Walk_N:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_NE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_E:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_SE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
            case DiabloEnv.ActionEnum.Walk_S:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_SW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_W:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_NW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP)
            case DiabloEnv.ActionEnum.Stand:
                # No key - NOP
                key = 0
            case DiabloEnv.ActionEnum.PrimaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_A)
            case DiabloEnv.ActionEnum.SecondaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_B)
        return key

    # Return neighbours in cardinal CW directions:
    # N, NE, E, SE, S, SW, W, NW
    @staticmethod
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

    @staticmethod
    def action_mask(d, env, what):
        """Computes an action mask for the action space using the state information."""
        mask = np.full(len(DiabloEnv.ActionEnum), 1, dtype=np.int8)
        player_pos = [d.player.position.future.x, d.player.position.future.y]

        neighbors = DiabloEnv.get_matrix_neighbors(env, *player_pos)
        for i, tile in enumerate(neighbors):
            if tile < 0:
                # Block in the direction beyond map
                mask[i] = 0
                continue

            if what & DiabloEnv.ActionMask.MASK_TRIGGERS.value and \
               tile & diablo_state.EnvironmentFlag.Trigger.value:
                # Block in the direction of any trigger, so forbid
                # escaping to the town
                # TODO: this blocks all the trigger including stairs to the
                # next level
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_WALLS.value and \
                 tile & diablo_state.EnvironmentFlag.Wall.value:
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_CLOSED_DOORS.value and \
                 tile & diablo_state.EnvironmentFlag.DoorClosed.value:
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_OTHER_SOLIDS.value and \
                 (tile & diablo_state.EnvironmentFlag.Barrel.value or \
                  tile & diablo_state.EnvironmentFlag.ChestClosed.value or \
                  tile & diablo_state.EnvironmentFlag.ChestOpened.value or \
                  tile & diablo_state.EnvironmentFlag.SarcophClosed.value or \
                  tile & diablo_state.EnvironmentFlag.SarcophOpened.value):
                mask[i] = 0

        return mask

    def submit_key(self, key, wait):
        entry = self.game.state.input_queue.get_entry_to_submit()
        assert entry
        entry.type = key
        entry.data = 0

        # Submit key
        self.game.state.input_queue.submit()

        if wait & DiabloEnv.KeyWaitFlags.WAIT_KEY_ACCEPTANCE.value:
            # Busy-loop for actual key acceptance
            while self.game.state.input_queue.nr_submitted_entries() != 0:
                time.sleep(0.01)

        if wait & DiabloEnv.KeyWaitFlags.WAIT_FOR_NEXT_TICK.value:
            # Busy-loop for next tick after key was injected
            next_tick = self.game.state.game_tick + 1
            while self.game.state.game_tick < next_tick:
                time.sleep(0.01)

    def __init__(self, env_config):
        self.config = env_config
        self.seed = env_config["seed"] + env_config.worker_index
        self.paused = False

        # Run or attach to Diablo
        self.game = diablo_state.DiabloGame.run_or_attach(env_config)

        # Submit SAVE, so we LOAD on reset()
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.submit_key(key, DiabloEnv.WAIT_KEY_ACCEPTANCE_AND_NEXT_TICK)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d)

        self.action_space = gym.spaces.Discrete(len(DiabloEnv.ActionEnum))

        env_high = (1 << len(diablo_state.EnvironmentFlag)) - 1
        # Don't forget to change dtype if does not fit max number of bits
        assert env_high <= np.iinfo(env.dtype).max
        self.observation_space = gym.spaces.Dict({
                "env-status":  gym.spaces.Box(low=0,
                                              high=0xffffff,
                                              shape=env_status.shape,
                                              dtype=env_status.dtype),
                "environment": gym.spaces.Box(low=0,
                                              high=env_high,
                                              shape=env.shape,
                                              dtype=env.dtype),
            }
        )

    def pause_game(self, pause=True):
        if self.paused != pause:
            self.paused = pause
            key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE | \
                ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            self.submit_key(key, DiabloEnv.WAIT_KEY_ACCEPTANCE_AND_NEXT_TICK)

    def close(self):
        self.game.stop_or_detach()

    def get_env_status(self, d):
        kills = np.sum(d.MonsterKillCounts_np)
        hp = d.player._pHitPoints
        mode = d.player._pmode
        return np.array([kills, hp, mode])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)

        # Submit LOAD
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.submit_key(key, DiabloEnv.WAIT_KEY_ACCEPTANCE_AND_NEXT_TICK)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d)

        obj_cnt = diablo_state.count_active_objects(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
        hp = d.player._pHitPoints

        self.prev_obj_cnt = obj_cnt
        self.prev_items_cnt = items_cnt
        self.prev_monsters_cnt = monsters_cnt
        self.prev_total_hp = total_hp
        self.prev_explored_cnt = explored_cnt
        self.prev_hp = hp

        init_obs = {
            "env-status":  env_status,
            "environment": env,
        }

        # Probability is always 1.0, diablo environment is deterministic
        init_info = {
            "prob": 1.0,
            "action_mask": DiabloEnv.action_mask(d, env,
                                   DiabloEnv.ActionMask.MASK_TRIGGERS.value)
        }

        return init_obs, init_info

    def evaluate_step(self, d, env, action):
        # truncated flag indicates whether the episode was forcefully
        # stopped due to a time limit or other constraints not related
        # to task completion.
        truncated = False
        done = False

        action_mask = DiabloEnv.action_mask(d, env,
                               DiabloEnv.ActionMask.MASK_TRIGGERS.value)

        obj_cnt = diablo_state.count_active_objects(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
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
        # 2.  reward for staying at high health

        # Strategic Encouragement
        # +1. reward for collecting items
        # 2. penalty for attacking walls or wasting actions
        # 3. reward for reaching the next level stairs after full explorations

        # Endgame conditions
        # +1. Huge penalty for dying
        # 2. Huge penalty for escaping to the town
        # 3. Huge penalty for descending if current dungeon is not cleared


        #### ITEMS PICK:   ON PRESS
        #### DOORS OPEN:   ON PRESS
        #### OBJECTS OPEN: ON RELEASE
        ####  PM_ATTACK:   ON PRESS
        #### TODO: handle PM_GOTHIT, PM_ATTACK
        ####

        # Default penalty for NOP
        reward = -0.1

        if action_mask[action] == 0:
            # XXX: That's a cludge: I do not know how to make an
            # XXX: action_mask for RLlib work, so we just done with this
            # XXX: episode with negative reward if agent steps into the
            # XXX: trigger.
            reward = -100
            done = True
        if hp < self.prev_hp:
            if hp <= 0:
                # We are dead, game over
                reward = -1000
                done = True
            else:
                # More damage to health - more penalty
                # from 0 to -100
                change = (hp - self.prev_hp) / self.prev_hp
                reward = change * 100
            self.prev_hp = hp
        if explored_cnt > self.prev_explored_cnt:
            # Exploration
            reward = 1
            self.prev_explored_cnt = explored_cnt
        if obj_cnt < self.prev_obj_cnt:
            # Reward for opening doors, chests, etc
            reward = 5
            self.prev_obj_cnt = obj_cnt
        if items_cnt < self.prev_items_cnt:
            # Reward for collecting items. Item number can increase if
            # dropped from a chest, but actions can't be applied on a
            # closed chest and picking up an item, so object counter
            # should be checked first
            reward = 5
            self.prev_items_cnt = items_cnt
        if total_hp < self.prev_total_hp:
            # Monster took damage
            reward = 10
            self.prev_total_hp = total_hp
        if monsters_cnt < self.prev_monsters_cnt:
            # Monsters killed
            reward = (self.prev_monsters_cnt - monsters_cnt) * 20
            self.prev_monsters_cnt = monsters_cnt

        return reward, done, truncated, action_mask

    def step(self, action):
        if self.paused:
            self.pause_game(False)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d)

        reward, done, truncated, action_mask = self.evaluate_step(d, env, action)

        if not done:
            # Submit an action key only if we have not completed this
            # episode.  The reason is as follows: if we escape to the
            # town, then LOAD does not always work. I assume this is
            # because there are many transitional states in Diablo
            # itself and the LOAD is simply ignored. The easiest fix
            # is not to send a key at all because we are done anyway
            # (either by death or by escaping using a trigger).
            #
            # TODO: Escaping to town should probably be reflected in
            # the observation state, otherwise agent is not able to
            # observe why reward became negative.
            key = DiabloEnv.action_to_key(action)
            key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            self.submit_key(key, DiabloEnv.DONT_WAIT)

        next_obs = {
            "env-status":  env_status,
            "environment": env,
        }
        # Probability is always 1.0, diablo environment is deterministic
        next_info = {
            "prob": 1.0,
            "action_mask": action_mask
        }

        return next_obs, reward, done, truncated, next_info
