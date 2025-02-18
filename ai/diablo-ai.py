#!/usr/bin/env python3

"""Diablo AI tool

   Tool which trains AI for playing Diablo and helps to evalulate and
   play as human

Usage:
  diablo-ai.py play     [--attach=MEM_PATH_OR_PID] [--no-monsters] [--no-env-log] [--seed=SEED]
  diablo-ai.py play-ai  [--attach=MEM_PATH_OR_PID] [--no-monsters] [--seed=SEED]
  diablo-ai.py train-ai [--attach=MEM_PATH_OR_PID] [--no-monsters] [--seed=SEED] [--same-seed] [--train-batch-size=SIZE] [--train-iters=ITERS] [--gpus=NR] [--env-runners=NR] [--tune] [--log-to-stdout] [--no-actions] [--restore-from-checkpoint] [--save-to-checkpoint] [--exploration-door-attraction] [--exploration-door-backtrack-penalty]
  diablo-ai.py list
  diablo-ai.py (-h | --help)
  diablo-ai.py --version

Modes:
  play           Let the human play Diablo or attach to an existing Diablo instance (devilutionX process) by providing the --attach option.
  play-ai        Let AI play Diablo.
  train-ai       Train the AI by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by providing the --attach option (convenient for debug purposes).
  list           List all Diablo instances (devilutionX processes).

Options:
  -h --help
      Show this screen.

  --version
      Show version.

  --attach=MEM_PATH_OR_PID
      Attach to existing Diablo instance by path, pid or  an index of an instance from the `diablo-ai.py list` output. For example:
        Attach by PID:
           diablo-ai.py play --attach 112342

         Attach by path:
           diablo-ai.py play --attach /tmp/diablo-tj3bxyvy/shared.mem

         Attach by index:
           diablo-ai.py play --attach 0

  --no-monsters
    Disable all monsters on the level.

  --no-env-log
      No environment log is shown on the TUI screen, which can speed up refresh rate on slow terminals (default: enabled, meaning that if a log is produced by the Gymnasium environment, it will be displayed).

  --seed=SEED
      Initial seed (default: 0).

  --same-seed
      Set same seed for all runners (default: seed += worker_index - 1)

  --train-batch-size=SIZE
      Size of a train batch (default: 200)

  --train-iters=ITERS
      Number of train iterations (default: 3)

  --gpus=NR
      Number of GPUs (default: 0)

  --env-runners=NR
      Number of environment runners (default: 1)

  --tune
      Run hyperparameters tuner first

  --log-to-stdout
      Write logs to stdout (default: env.log file in Diablo state folder)

  --no-actions
      Disable agent from generating actions. Very handy mode to attach and play manually to simulate reward (default: actions enabled)

  --restore-from-checkpoint
      Restores state from a checkpoint prior training (default: no restore from a checkpoint)

  --save-to-checkpoint
      Saves state to a checkpoint on each train loop (default: no save to a checkpoint)

  --exploration-door-attraction
      Reward increases when the agent moves closer to unexplored doors. Helps guide the agent toward new areas faster (default: disabled).

  --exploration-door-backtrack-penalty
      Penalizes the agent for moving away from unexplored doors after approaching them. Encourages committing to exploration paths instead of turning back. If set, enables the `--exploration-door-attraction`.
"""
VERSION='Diablo AI Tool v1.0'

from docopt import docopt
from pathlib import Path
import collections
import configparser
import copy
import curses
import enum
import numpy as np
import os
import psutil
import re
import subprocess
import sys
import tempfile
import time

import diablo_env
import diablo_state
import ring


# Deactivate the new API stack and switch back to the old one.
# New stack creates several environments (why???), which means we have
# one Diablo process idling. Also train results for old stack include
# reward.
NEW_API_STACK=False

# Flag to control the main loop
running = True
# Global variable to track the last key pressed
last_key = 0
# Global last tick
last_tick = 0

# This is weird, but if you place a character in the last column,
# curses fills that position, yet still raises an error.
# These two wrappers attempt to ignore an error if it occurs
# when filling in the last position.
def _addstr(o, y, x, text):
    try:
        return o.addstr(y, x, text)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise
        pass

# See the comment for the _addstr
def _addch(o, y, x, ch):
    try:
        return o.addch(y, x, ch)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise
        pass

def truncate_line(line, N, extra='...'):
    if N <= len(extra):
        return ""
    return line[:N-len(extra)] + extra if len(line) > N else line

class EnvLog:
    fd = None
    queue = None

    def __init__(self, fd):
        self.fd = fd

def open_envlog(game):
    path = os.path.join(os.path.dirname(game.mshared_path), "env.log")
    fd = None
    try:
        fd = open(path, "r")
        return EnvLog(fd)
    except:
        pass
    return None

def display_env_log(game, stdscr, envlog):
    if envlog is None:
        return

    height, width = stdscr.getmaxyx()
    logwin_h = height // 2
    logwin_w = width // 4

    h = max(0, logwin_h - 2)
    w = max(0, logwin_w - 2)

    # Sane limitation
    if h < 10 or w < 20:
        return

    logwin = stdscr.subwin(logwin_h, logwin_w, 4, 1)

    if envlog.queue is None:
        queue = collections.deque(maxlen=h)
    elif envlog.queue.maxlen != logwin_h:
        queue = collections.deque(maxlen=h)
        for line in envlog.queue:
            queue.append(line)
    else:
        queue = envlog.queue

    while True:
        line = envlog.fd.readline()
        if not line:
            break
        queue.append(line)

    logwin.clear()
    logwin.border()
    msg = " Environment log "
    _addstr(logwin, 0, w//2 - len(msg)//2, msg)
    for i, line in enumerate(queue):
        line = truncate_line(line.strip(), w)
        _addstr(logwin, i+1, 1, line)
    logwin.refresh()

    envlog.queue = queue


def get_processes_by_binary(binary_path):
    """Find all processes matching the specified command binary name."""
    matching_processes = []
    for proc in psutil.process_iter(attrs=['pid', 'name', 'exe', 'cmdline']):
        if binary_path == proc.info['exe']:
            matching_processes.append(proc.info)
    return matching_processes

def get_mapped_files_of_pid(pid):
    """Get a list of memory-mapped files from /proc/<pid>/maps."""
    mapped_files = []
    maps_path = f"/proc/{pid}/maps"

    if os.path.exists(maps_path):
        try:
            with open(maps_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    # Last part is a file path
                    if len(parts) > 5 and parts[-1].startswith('/'):
                        mapped_files.append(parts[-1])
        except (OSError, IOError):
            # Ignore unreadable files
            pass

    return mapped_files

def get_mapped_file_of_pid(pid, filename):
    mapped_files = get_mapped_files_of_pid(pid)

    # Match only the filename but keep the full path
    matching_files = [f for f in mapped_files if os.path.basename(f) == filename]

    if matching_files:
        return matching_files[0]

    return None

def procs_natural_sort(p, _nsre=re.compile(r'(\d+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(p['mshared_path'])]

def find_processes_with_mapped_file(binary_path, filename):
    """Find processes with mapped filename"""
    matching_processes = get_processes_by_binary(binary_path)
    result = []

    for proc in matching_processes:
        pid = proc['pid']
        mshared_path = get_mapped_file_of_pid(pid, filename)

        if mshared_path:
            result.append({'pid': pid,
                           'exe': proc['exe'],
                           'cmdline': proc['cmdline'],
                           'mshared_path': mshared_path})

    return sorted(result, key=procs_natural_sort)

def list_devilution_processes(binary_path, mshared_filename):
    result = find_processes_with_mapped_file(binary_path, mshared_filename)
    if result:
        for proc in result:
            print("%s\t%s" % (proc['pid'], proc['mshared_path']))

def handle_keyboard(stdscr):
    global last_key
    global running

    k = stdscr.getch()
    if k == -1:
        return False

    key = 0

    if k == 259:
        key = ring.RingEntryType.RING_ENTRY_KEY_UP
    elif k == 258:
        key = ring.RingEntryType.RING_ENTRY_KEY_DOWN
    elif k == 260:
        key = ring.RingEntryType.RING_ENTRY_KEY_LEFT
    elif k == 261:
        key = ring.RingEntryType.RING_ENTRY_KEY_RIGHT
    elif k == ord('a'):
        key = ring.RingEntryType.RING_ENTRY_KEY_A
    elif k == ord('b'):
        key = ring.RingEntryType.RING_ENTRY_KEY_B
    elif k == ord('x'):
        key = ring.RingEntryType.RING_ENTRY_KEY_X
    elif k == ord('y'):
        key = ring.RingEntryType.RING_ENTRY_KEY_Y
    elif k == ord('l'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == ord('s'):
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE
    elif k == ord('p'):
        key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE
    elif k == ord('q'):
        running = False  # Stop the main loop

    last_key |= key

    return True

def remap_movement_keys(keys):
    # The dungeon in Diablo is rotated 45 degrees clockwise. To
    # compensate for this rotation and make UP true North, rather than
    # a diagonal movement, we should send two keys for each direction.

    movement_bits = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                     ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                     ring.RingEntryType.RING_ENTRY_KEY_LEFT |
                     ring.RingEntryType.RING_ENTRY_KEY_RIGHT)

    # Copy except movement bits
    reskeys = ~movement_bits & keys
    movement_keys = movement_bits & keys

    if movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP):
        # N
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP |
                    ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                           ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # NE
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # E
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                    ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                           ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # SE
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN):
        # S
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                    ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                           ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # SW
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # W
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP |
                    ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                           ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # NW
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP)

    return reskeys

def get_radius(d, dunwin):
    height, width = dunwin.getmaxyx()
    width = min(width, d.maxdun[0])
    height = min(height, d.maxdun[1])
    return np.array([width // 2, height // 2])

def display_matrix(dunwin, m):
    cols, rows = m.shape

    # Get the screen size
    height, width = dunwin.getmaxyx()

    x_off = width // 2 - cols // 2
    y_off = height // 2 - rows // 2

    assert(x_off >= 0)
    assert(y_off >= 0)

    for row in range(rows):
        for col in range(cols):
            _addch(dunwin, row + y_off, col + x_off, m[col, row])

def display_dungeon(d, stdscr):
    height, width = stdscr.getmaxyx()
    dunwin = stdscr.subwin(height - (4 + 1), width, 4, 0)
    radius = get_radius(d, dunwin)
    surroundings = diablo_state.get_surroundings(d, radius)

    display_matrix(dunwin, surroundings)

def display_diablo_state(game, stdscr, envlog_fd, missed_ticks):
    # Unfortunately (performance-wise) we have to make a deep copy to
    # prevent partial or complete changes of the state in the middle
    # of this routine
    d = copy.deepcopy(game.state)
    pos = diablo_state.player_position(d)

    # Get the screen size
    height, width = stdscr.getmaxyx()

    msg = "Diablo ticks: %d (missed: %d); Kills: %003d; HP: %d; Pos: %d:%d; State: %-18s" % \
        (game.ticks(d),
         # Always 1 ticks behind
         missed_ticks - 1,
         np.sum(d.MonsterKillCounts_np),
         d.player._pHitPoints,
         pos[0], pos[1],
         diablo_state.PLR_MODE(d.player._pmode).name)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 0, width // 2 - len(msg) // 2, msg)

    msg = "Press 'q' to quit"
    _addstr(stdscr, height - 1, width // 2 - len(msg) // 2, msg)

    msg = "Animation: ticksPerFrame %2d; tickCntOfFrame %2d; frames %2d; frame %2d" % \
        (d.player.AnimInfo.ticksPerFrame,
         d.player.AnimInfo.tickCounterOfCurrentFrame,
         d.player.AnimInfo.numberOfFrames,
         d.player.AnimInfo.currentFrame)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 1, width // 2 - len(msg) // 2, msg)

    obj_cnt = diablo_state.count_active_objects(d)
    items_cnt = diablo_state.count_active_items(d)
    total_hp = diablo_state.count_active_monsters_total_hp(d)

    msg = "Total: monsters HP %d, items %d, objects %d, Dungeon: %d" % \
        (total_hp, items_cnt, obj_cnt, d.player.plrlevel)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 2, width // 2 - len(msg) // 2, msg)

    display_dungeon(d, stdscr)
    display_env_log(game, stdscr, envlog_fd)

    if diablo_state.is_game_paused(d):
        msgs = ["            ",
                " ┌────────┐ ",
                " │ Paused │ ",
                " └────────┘ ",
                "            "]
        h = height // 2
        for i, msg in enumerate(msgs):
            _addstr(stdscr, h + i, width // 2 - len(msg) // 2, msg)


def run_tui(stdscr, gameconfig):
    global running
    global last_key

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(True)

    missed_ticks = 0
    envlog = None

    # Main loop
    while running:
        stdscr.clear()

        if not gameconfig['--no-env-log'] and envlog is None:
            # Try to open a environment log, can be creater later
            envlog = open_envlog(game)

        display_diablo_state(game, stdscr, envlog, missed_ticks)

        if last_key:
            # Compensate dungeon 45CW rotation
            key = remap_movement_keys(last_key)
            key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            game.submit_key(key)
            last_key = 0

        # Refresh the screen to show the content
        stdscr.refresh()

        # Handle keys and introduce delay until next tick
        while running and (handle_keyboard(stdscr) or game.same_ticks()):
            pass

        missed_ticks = game.update_ticks()

def ai(args, gameconfig):
    import torch

    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.utils.metrics import (
        ENV_RUNNER_RESULTS,
        EPISODE_RETURN_MEAN,
        EPISODE_LEN_MEAN,
    )
    import ray
    from ray import air, tune, train
    from ray.air.constants import TRAINING_ITERATION
    from ray.rllib.algorithms.ppo import PPO
    from ray.tune.registry import register_env
    from ray.tune.schedulers import create_scheduler
    from ray.rllib.utils.test_utils import check_learning_achieved

    class DiabloEnvCallback(RLlibCallback):
        # New API
        #def on_sample_end(self, *args, env_runner, metrics_logger, **kwargs):
        def on_sample_end(self, worker, samples):
            """Pause Diablo environment."""
            worker.env.pause_game(True)

    # Register Diablo Gym environment
    register_env("diablo", lambda cfg: diablo_env.DiabloEnv(cfg))

    # Prepare config
    config = (
        PPOConfig()
        .environment("diablo", env_config=gameconfig)
        .resources(
            num_gpus=args['--gpus'],
#            num_cpus_per_worker=1,
#            num_gpus_per_worker=0,
        )
        .debugging(log_level='ERROR') # INFO, DEBUG, ERROR, WARN
        .framework('torch')
        .env_runners(
            num_envs_per_env_runner=1,
            num_env_runners=args['--env-runners'] if args['train-ai'] else 0,
            # Observations are discrete (ints) -> We need to flatten (one-hot) them.
            env_to_module_connector=lambda env: FlattenObservations(),
        )
        .callbacks(DiabloEnvCallback)
        .training(
            train_batch_size=args['--train-batch-size'],
            lr=tune.grid_search([0.01, 0.005, 0.003, 0.001, 0.0001])
            if args['--tune'] else 0.001
        )
        .evaluation(evaluation_num_env_runners=0)
        .api_stack(enable_rl_module_and_learner=NEW_API_STACK,
                   enable_env_runner_and_connector_v2=NEW_API_STACK)
    )

    if args['train-ai'] and args['--tune']:
        print("RUN TUNE")

        # ensure that checkpointing works.
        pbt = create_scheduler(
            "pbt",
            perturbation_interval=1,  # To make perturb more often.
            hyperparam_mutations={
                "train_loop_config": {
                    "lr": config.lr
                },
            },
        )

        # Get the best checkpoints from the trial, based on different metrics.
        # Checkpoint with the lowest policy loss value:
        if NEW_API_STACK:
            policy_loss_key = f"{LEARNER_RESULTS}/{DEFAULT_MODULE_ID}/policy_loss"
        else:
            policy_loss_key = "info/learner/default_policy/learner_stats/policy_loss"

        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=train.RunConfig(
                #verbose=1,
                stop={TRAINING_ITERATION: 1},
            ),
            tune_config=tune.TuneConfig(
                num_samples=1,
                metric=policy_loss_key,
                mode="min",
                scheduler=pbt,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()
        best_config = best_result.config

        pl = best_result.metrics["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]

        print("TUNE FINISHED: lr=%f, policy_loss=%f" % (best_config['lr'], pl))

        config.lr = best_config['lr']

    algo = config.build()

    CHECKPOINT_PATH = os.path.abspath("./diablo.ppo.checkpoint")

    if not args['train-ai'] or \
       args['--restore-from-checkpoint'] and os.path.exists(CHECKPOINT_PATH):
        # Restore for play-ai mode or if told explicitly
        algo.restore(CHECKPOINT_PATH)

    if args['train-ai']:
        for i in range(args['--train-iters']):
            results = algo.train()
            if args['--save-to-checkpoint']:
                algo.save(CHECKPOINT_PATH)

            reward = results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
            len_mean = results[ENV_RUNNER_RESULTS][EPISODE_LEN_MEAN]

            print("Finished iter %d/%d; avg. reward %.2f, episode_len_mean=%.2f" %
                  (i+1, args['--train-iters'], reward, len_mean))

        # Stop the algorithm. Note, this is important for when
        # defining `output_max_rows_per_file`. Otherwise,
        # remaining episodes in the `EnvRunner`s buffer isn't written to disk.
        algo.stop()
    else:
        # XXX This is a nasty kludge to prevent RLlib from spawning an extra runner
        # XXX for training, even though we do evaluation. The `self-evaluation`
        # XXX flag will be checked in the diablo_env.
        gameconfig['self-evaluation'] = True

        env = diablo_env.DiabloEnv(gameconfig)
        # Get the initial observation
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print("Finished 1 episode, total-reward=%.3f" % (total_reward))

def main():
    args = docopt(__doc__, version=VERSION)

    args["--seed"] = 0 if args["--seed"] is None \
        else int(args["--seed"])
    args["--train-batch-size"] = 200 if args["--train-batch-size"] is None \
        else int(args["--train-batch-size"])
    args["--train-iters"] = 3 if args["--train-iters"] is None \
        else int(args["--train-iters"])
    args["--env-runners"] = 1 if args["--env-runners"] is None \
        else int(args["--env-runners"])
    args["--gpus"] = 0 if args["--gpus"] is None \
        else int(args["--gpus"])
    args["--exploration-door-attraction"] = True if args["--exploration-door-backtrack-penalty"] \
        else args["--exploration-door-attraction"]

    config = configparser.ConfigParser()
    config.read('diablo-ai.ini')

    diablo_bin_path  = Path(config['default']['diablo-bin-path'])
    diablo_data_path = Path(config['default']['diablo-data-path'])
    diablo_mshared_filename = config['default']['diablo-mshared-filename']

    if not diablo_bin_path.exists() or not diablo_data_path.exists() or \
       len(diablo_mshared_filename) == 0:
        print("Error: initial configuration is invalid. Please check your 'diablo-ai.ini' file and provide valid paths for 'diablo-bin-path', 'diablo-data-path' and 'diablo-mshared-filename' configuration options.")
        sys.exit(1)

    gameconfig = {
        "seed": args["--seed"],
        "headless": 1,
        "mshared_filename": diablo_mshared_filename,
        "no_monsters": 1 if args["--no-monsters"] else 0,

        "train-ai": args["train-ai"],
        "--log-to-stdout": args["--log-to-stdout"],
        "--no-actions": args["--no-actions"],
        "--no-env-log": args["--no-env-log"],
        "--same-seed": args["--same-seed"],
        "--exploration-door-attraction": args["--exploration-door-attraction"],
        "--exploration-door-backtrack-penalty": args["--exploration-door-backtrack-penalty"],
    }

    if args['--attach']:
        path_or_pid = args['--attach']

        if re.match(r'^\d+$', path_or_pid):
            pid_or_index = int(path_or_pid)
            procs = find_processes_with_mapped_file(str(diablo_bin_path),
                                                    diablo_mshared_filename)
            if pid_or_index < len(procs):
                gameconfig['attach-path'] = procs[pid_or_index]['mshared_path']
            else:
                mshared_path = get_mapped_file_of_pid(pid_or_index,
                                                      diablo_mshared_filename)
                if mshared_path:
                    gameconfig['attach-path'] = mshared_path
        elif os.path.exists(path_or_pid):
            gameconfig['attach-path'] = path_or_pid

        if 'attach-path' not in gameconfig:
            print("Error: --attach=%s is not a valid path, PID or index of a Diablo instance" %
                  path_or_pid)
            sys.exit(1)
    else:
        gameconfig['diablo-bin-path']  = diablo_bin_path
        gameconfig['diablo-data-path'] = diablo_data_path

    if args['play']:
        # Run the curses application
        curses.wrapper(lambda stdscr: run_tui(stdscr, gameconfig))
    elif args['train-ai']:
        ai(args, gameconfig)
    elif args['play-ai']:
        ai(args, gameconfig)
    elif args['list']:
        list_devilution_processes(str(diablo_bin_path),
                                  diablo_mshared_filename)
    else:
        print("Not supported yet")

if __name__ == "__main__":
    main()
