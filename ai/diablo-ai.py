#!/usr/bin/env python3

"""Diablo AI tool

   Tool which trains AI for playing Diablo and helps to evalulate and
   play as human

Usage:
  diablo-ai.py play     [--attach=MEM_PATH_OR_PID]
  diablo-ai.py play-ai  [--attach=MEM_PATH_OR_PID]
  diablo-ai.py train-ai [--attach=MEM_PATH_OR_PID]
  diablo-ai.py list
  diablo-ai.py (-h | --help)
  diablo-ai.py --version

Modes:
  play           Let human play Diablo
     tui         Starts only text UI
     tui-and-gui Starts text UI and Diablo GUI in a separate window
  play-ai        Let AI play Diablo
  train-ai       Train AI
  list           List all Diablo instances (devilutionX processes)

Options:
  -h --help                  Show this screen.
  --version                  Show version.
  --attach=MEM_PATH_OR_PID   Attach to existing Diablo instance by path or pid
"""
VERSION='Diablo AI Tool v1.0'

from docopt import docopt
from pathlib import Path
from pynput import keyboard
import configparser
import copy
import curses
import enum
import gc
import os
import psutil
import subprocess
import sys
import tempfile
import time

import diablo_gym
import diablo_state
import ring

import numpy as np

# Flag to control the main loop
running = True

# Global variable to track the last key pressed
prev_key = 0
last_key = 0

# Global last tick
last_tick = 0

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

def find_processes_with_mapped_file(binary_path, filename):
    """Find processes with mapped filename"""
    matching_processes = get_processes_by_binary(binary_path)
    result_list = []

    for proc in matching_processes:
        pid = proc['pid']
        mshared_path = get_mapped_file_of_pid(pid, filename)

        if mshared_path:
            result_list.append({'pid': pid,
                                'exe': proc['exe'],
                                'cmdline': proc['cmdline'],
                                'mshared_path': mshared_path})

    return result_list

def list_devilution_processes(binary_path, mshared_filename):
    result = find_processes_with_mapped_file(binary_path, mshared_filename)
    if result:
        for proc in result:
            print("%s\t%s" % (proc['pid'], proc['mshared_path']))

def on_key(press, k):
    global last_key
    global running

    key = 0

    if k == keyboard.Key.up:
        key = ring.RingEntryType.RING_ENTRY_KEY_UP
    elif k == keyboard.Key.down:
        key = ring.RingEntryType.RING_ENTRY_KEY_DOWN
    elif k == keyboard.Key.left:
        key = ring.RingEntryType.RING_ENTRY_KEY_LEFT
    elif k == keyboard.Key.right:
        key = ring.RingEntryType.RING_ENTRY_KEY_RIGHT
    elif k == keyboard.KeyCode.from_char('a'):
        key = ring.RingEntryType.RING_ENTRY_KEY_A
    elif k == keyboard.KeyCode.from_char('b'):
        key = ring.RingEntryType.RING_ENTRY_KEY_B
    elif k == keyboard.KeyCode.from_char('x'):
        key = ring.RingEntryType.RING_ENTRY_KEY_X
    elif k == keyboard.KeyCode.from_char('y'):
        key = ring.RingEntryType.RING_ENTRY_KEY_Y
    elif k == keyboard.KeyCode.from_char('l'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == keyboard.KeyCode.from_char('s'):
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE
    elif (k == keyboard.Key.esc or k == keyboard.KeyCode.from_char('q')) \
         and press:
        running = False  # Stop the main loop

    if key != 0:
        if press:
            last_key |= key
        else:
            last_key &= ~key

def on_key_press(key):
    on_key(True, key)

def on_key_release(key):
    on_key(False, key)

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

def get_radius(d, stdscr):
    height, width = stdscr.getmaxyx()
    width = min(width, d.maxdun[0])
    height = min(height, d.maxdun[1])
    return (width // 2, height // 2 - 3)

def display_matrix(stdscr, m):
    rows, cols = m.shape

    # Get the screen size
    height, width = stdscr.getmaxyx()

    x_off = width // 2 - cols // 2
    y_off = 4

    for row in range(rows):
        for col in range(cols):
            stdscr.addstr(row + y_off, col + x_off, "%s" % m[row, col])

def update_ticks(d):
    global last_tick
    missed = d.game_tick - last_tick
    last_tick += missed
    return missed

def same_ticks(d):
    global last_tick
    diff = d.game_tick - last_tick
    return diff == 0

def display_diablo_state(stdscr, d_shared, missed_ticks):
    # Unfortunately (performance-wise) we have to make a deep copy to
    # prevent partial or complete changes of the state in the middle
    # of this routine
    d = copy.deepcopy(d_shared)

    # Get the screen size
    height, width = stdscr.getmaxyx()

    hdr_msg = "Diablo ticks: %d (missed: %d); Kills: %003d; HP: %d; Pos: %d:%d; State: %-18s" % \
        (d.game_tick,
         # Always 1 ticks behind
         missed_ticks - 1,
         np.sum(d.MonsterKillCounts_np),
         d.player._pHitPoints,
         d.player.position.future.x, d.player.position.future.y,
         diablo_state.PLR_MODE(d.player._pmode).name)
    quit_msg = "Press 'ESC' to quit"

    stdscr.addstr(0, width // 2 - len(hdr_msg) // 2, hdr_msg)
    stdscr.addstr(height - 1, width // 2 - len(quit_msg) // 2, quit_msg)

    msg = "Animation: ticksPerFrame %d; tickCntOfFrame %d; frames %d; frame %d" % \
        (d.player.AnimInfo.ticksPerFrame,
         d.player.AnimInfo.tickCounterOfCurrentFrame,
         d.player.AnimInfo.numberOfFrames,
         d.player.AnimInfo.currentFrame)

    stdscr.addstr(1, width // 2 - len(msg) // 2, msg)

    obj_cnt = diablo_state.count_active_objects(d)
    items_cnt = diablo_state.count_active_items(d)
    total_hp = diablo_state.count_active_monsters_total_hp(d)

    msg = "Total: monsters HP %d, items %d, objects %d" % \
        (total_hp, items_cnt, obj_cnt)

    stdscr.addstr(2, width // 2 - len(msg) // 2, msg)

    radius = get_radius(d, stdscr)
    dun_rect = diablo_state.DungeonRect(d=d, radius=radius)

    surroundings = diablo_state.get_surroundings(d, dun_rect)

    display_matrix(stdscr, surroundings)

def run_tui(stdscr, gameconfig):
    global running
    global new_key
    global prev_key

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # Start the keyboard listener in non-blocking mode
    keylistener = keyboard.Listener(on_press=on_key_press,
                                    on_release=on_key_release)
    keylistener.start()

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(1)

    missed_ticks = 0

    # Main loop
    while running:
        # Clear the screen
        stdscr.clear()

        display_diablo_state(stdscr, game.state, missed_ticks)

        new_key = last_key
        if prev_key != new_key:
            prev_key = new_key

            # Get an entry to submit
            entry = game.state.input_queue.get_entry_to_submit()
            if entry:
                # Compensate dungeon 45CW rotation
                entry.type = remap_movement_keys(new_key)
                entry.data = 0
                game.state.input_queue.submit()

        # Refresh the screen to show the content
        stdscr.refresh()

        # Add a delay
        while running and same_ticks(game.state):
            pass

        missed_ticks = update_ticks(game.state)

    # Wait for listener to stop
    keylistener.stop()

def train_ai(args, gameconfig):
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
    from ray.tune.logger import pretty_print

    import ray
    from ray.rllib.algorithms.ppo import PPO

    from ray.tune.registry import register_env

    class DiabloEnvCallback(RLlibCallback):
        # New API
        #def on_sample_end(self, *args, env_runner, metrics_logger, **kwargs):
        def on_sample_end(self, worker, samples):
            """Pause Diablo environment."""
            worker.env.pause_game(True)

    # Register Diablo Gym environment
    register_env("diablo", lambda cfg: diablo_gym.DiabloEnv(cfg))

    # Prepare config
    config = (
        PPOConfig()
        .environment("diablo", env_config=gameconfig)
        # Disable GPU
        .resources(
            num_gpus=0,
#            num_cpus_per_worker=1,
#            num_gpus_per_worker=0,
        )
        .debugging(log_level='ERROR') # INFO, DEBUG, ERROR, WARN
        .framework('torch')
        .env_runners(
            num_envs_per_env_runner=1,
            num_env_runners=1,
            # Observations are discrete (ints) -> We need to flatten (one-hot) them.
            env_to_module_connector=lambda env: FlattenObservations(),
        )
        .callbacks(DiabloEnvCallback)
        .training(
            train_batch_size=200,
        )
#        .evaluation(evaluation_num_env_runners=0)
        # Deactivate the new API stack and switch back to the old one.
        # New stack creates several environments (why???), which means we have
        # one Diablo process idling. Also train results for old stack include
        # reward.
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
    )

    algo = config.build()

    CHECKPOINT_PATH = os.path.abspath("./diablo.ppo.checkpoint")

    #algo.restore(CHECKPOINT_PATH)

    for i in range(200):
        results = algo.train()
        algo.save(CHECKPOINT_PATH)

#        print(results)

        reward=results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        episode_len_mean = results[ENV_RUNNER_RESULTS][EPISODE_LEN_MEAN]

        print("Iter: %d; avg. reward %.2f, episode_len_mean=%.2f\n" %
              (i, reward, episode_len_mean))

    # Stop the algorithm. Note, this is important for when
    # defining `output_max_rows_per_file`. Otherwise,
    # remaining episodes in the `EnvRunner`s buffer isn't written to disk.
    algo.stop()

def main():
    args = docopt(__doc__, version=VERSION)

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
        "seed":     0,
        "headless": 1,
        "mshared_filename": diablo_mshared_filename
    }

    if args['--attach']:
        path_or_pid = args['--attach']
        if os.path.exists(path_or_pid):
            gameconfig['attach-path'] = path_or_pid
        else:
            mshared_path = get_mapped_file_of_pid(path_or_pid, diablo_mshared_filename)
            if not mshared_path:
                print("Error: --attach=%s is not a valid path or a PID of a Diablo instance" %
                      path_or_pid)
                sys.exit(1)
            gameconfig['attach-path'] = mshared_path
    else:
        gameconfig['diablo-bin-path']  = diablo_bin_path
        gameconfig['diablo-data-path'] = diablo_data_path

    if args['play']:
        # Run the curses application
        curses.wrapper(lambda stdscr: run_tui(stdscr, gameconfig))
    elif args['train-ai']:
        train_ai(args, gameconfig)
    elif args['list']:
        list_devilution_processes(str(diablo_bin_path),
                                  diablo_mshared_filename)
    else:
        print("Not supported yet")

if __name__ == "__main__":
    main()
