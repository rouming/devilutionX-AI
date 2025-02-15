"""
diablo_gym.py - Diablo Gymnasium environment
"""
import enum

import numpy as np
import gymnasium as gym

import diablo_state

class DiabloEnv(gym.Env):
    class ActionEnum(enum.Enum):
        Stand           = enum.auto()
        North           = enum.auto()
        NorthEast       = enum.auto()
        East            = enum.auto()
        SouthEast       = enum.auto()
        South           = enum.auto()
        SouthWest       = enum.auto()
        West            = enum.auto()
        NorthWest       = enum.auto()
        # Attack monsters, talk to towners, lift and place inventory items.
        PrimaryAction   = enum.auto()
        # Open chests, interact with doors, pick up items.
        SecondaryAction = enum.auto()

    def __init__(self, env_config):
        self.config = env_config
        self.seed = env_config["seed"] ^ env_config.worker_index

        cfg_file = open("cfg/diablo.ini.template", "r")
        cfg = cfg_file.read()
        cfg_file.close()
        cfg = cfg.format(seed=self.seed)

        prefix = "diablo-%d--" % env_config.worker_index
        self.state_dir = tempfile.TemporaryDirctory(prefix=prefix)
        cfg_file = open(self.state_dir.name + "/diablo.ini", "w")
        cfg_file.write(cfg)
        cfg_file.close()

        diablo_cmd = [
            env_config["diablo_bin"], '-n', '-f',
            '--config-dir', self.state_dir.name,
            '--save-dir', self.state_dir.name,
            '--data-dir', env_config["diablo_data_dir"]
        ]
        self.diablo_proc = subprocess.Popen(
            diablo_cmd,
            stdout=subprocess.DEVNULL,  # Ignore stdout
            stderr=subprocess.DEVNULL   # Ignore stderr
        )

        shared_mem_path = os.path.abspath(self.state_dir.name + "/shared.mem")
        for attempt in range(0, 10):
            try:
                # Open the file and map it to memory
                self.shared_file = open(shared_mem_path, "r+b")
                self.mmapped = mmap.mmap(shared_file.fileno(), 0)
                self.diablo = map_DiabloShared(mmapped)
            except FileNotFoundError:
                time.sleep(0.1)
        else:
            raise FileNotFoundError(shared_mem_path)

        # Submit SAVE
        entry = self.diablo.input_queue.get_entry_to_submit()
        assert entry
        entry.type = \
            ring.RingEntryType.RING_ENTRY_KEY_SAVE | \
            ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        entry.data = 0
        self.diablo.input_queue.submit()

        # Busy-loop for actual key acceptance
        while self.diablo.input_queue.nr_entries_to_submit() != \
              ring.RING_QUEUE_CAPACITY:
            time.sleep(0.01)

        self.action_space = spaces.Discrete(len(ActionEnum))
        self.observation_space = gym.spaces.Dict(
            {
                "game-state":  gym.spaces.Discrete(3),
                "environment": gym.spaces.Box(low=0, high=0xff,
                                              shape=(xx,xx),
                                              dtype=np.uint8),
            }
        )

    def get_game_state(self, d):
        kills = np.sum(d.MonsterKillCounts_np)
        hp = d.player._pHitPoints
        mode = d.player._pmode
        return np.array([kills, hp, mode])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)

        d = copy.deepcopy(self.diablo)
        obs = {
            "game-state":  self.get_game_state(d),
            "environment": self.get_environment(d),
        }
        # Probability is always 1.0, diablo environment is deterministic
        info = {"prob": 1.0, "action_mask": self.action_mask(d)}
        return obs, info

    def step(self, action):
        d = copy.deepcopy(self.diablo)
        obs = {
            "game-state":  self.get_game_state(d),
            "environment": self.get_environment(d),
        }
        reward = 1
        done = False
        # Flag indicates whether the episode was forcefully stopped
        # due to a time limit or other constraints not related to task
        # completion.
        truncated = False

        # Probability is always 1.0, diablo environment is deterministic
        info = {"prob": 1.0, "action_mask": self.action_mask(d)}
        return obs, reward, done, truncated, info

    def action_mask(self, d):
        """Computes an action mask for the action space using the state information."""
        mask = np.full(len(ActionEnum), 1, dtype=np.int8)

        # Forbid the way of the coward: never return to town
        # TODO: for now forbid all triggers
        for trig in d.trigs[:d.numtrigs]:
            dist = np.array([trig.position.x - d.player.position.future.x,
                             trig.position.y - d.player.position.future.y])
            if np.all(dist == (0, 1)):
                mask[ActionEnum.South.value] = 0
            elif np.all(dist == (1, 1)):
                mask[ActionEnum.SouthWest.value] = 0
            elif np.all(dist == (1, 0)):
                mask[ActionEnum.West.value] = 0
            elif np.all(dist == (1, -1)):
                mask[ActionEnum.NorthWest.value] = 0
            elif np.all(dist == (0, -1)):
                mask[ActionEnum.North.value] = 0
            elif np.all(dist == (-1, -1)):
                mask[ActionEnum.NorthEast.value] = 0
            elif np.all(dist == (-1, 0)):
                mask[ActionEnum.East.value] = 0
            elif np.all(dist == (-1, 1)):
                mask[ActionEnum.NorthWest.value] = 0
            break

        return mask
