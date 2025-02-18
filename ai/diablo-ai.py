#!/usr/bin/env python3

"""Diablo AI tool

   Tool which trains AI for playing Diablo and helps to evalulate and
   play as human

Usage:
  diablo-ai.py play     [--attach=MEM_PATH]
  diablo-ai.py play-ai  [--model=MODEL]
  diablo-ai.py train-ai [--model=MODEL]
  diablo-ai.py (-h | --help)
  diablo-ai.py --version

Modes:
  play           Let human play Diablo
     tui         Starts only text UI
     tui-and-gui Starts text UI and Diablo GUI in a separate window

  play-ai        Let AI play Diablo

  train-ai       Train AI

Options:
  -h --help          Show this screen.
  --version          Show version.
  --attach=MEM_PATH  Attach to existing Diablo instance
  --model=MODEL      AI model name
"""
VERSION='Diablo AI Tool v1.0'

from docopt import docopt
import configparser
import os
import curses
import time
import sys
from pynput import keyboard
import enum
import gc
import copy
import tempfile
import subprocess
from pathlib import Path

import ring
import diablo_state
import diablo_gym

import numpy as np

# Flag to control the main loop
running = True

# Global variable to track the last key pressed
prev_key = 0
last_key = 0

# Global last tick
last_tick = 0

def on_key(press, k):
    global last_key
    global running

    key = 0

    # Remember, map in diablo rotated 45 degrees CW.  To compensate
    # this rotation while moving and e.g. make UP real UP, and not a
    # diagonal movement we should send two keys for each direction.

    if k == keyboard.Key.up:
        key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
               ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif k == keyboard.Key.down:
        key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
               ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif k == keyboard.Key.left:
        key = (ring.RingEntryType.RING_ENTRY_KEY_LEFT |
               ring.RingEntryType.RING_ENTRY_KEY_UP)
    elif k == keyboard.Key.right:
        key = (ring.RingEntryType.RING_ENTRY_KEY_RIGHT |
               ring.RingEntryType.RING_ENTRY_KEY_DOWN)
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
    elif k == keyboard.Key.esc and press:
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

def run_tui(stdscr, game):
    global running
    global new_key
    global prev_key

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
                entry.type = new_key | \
                    ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
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

def main():
    arguments = docopt(__doc__, version=VERSION)

    if arguments['play']:
        if arguments['--attach']:
            game = diablo_state.DiabloGame.attach(arguments['--attach'])
        else:
            config = configparser.ConfigParser()
            config.read('diablo-ai.ini')

            diablo_bin_path  = Path(config['default']['diablo-bin-path'])
            diablo_data_path = Path(config['default']['diablo-data-path'])

            if not diablo_bin_path.exists() or not diablo_data_path.exists():
                print("Error: initial configuration is invalid. Please check your 'diablo-ai.ini' file and provide valid paths for 'diablo-bin-path' and 'diablo-data-path' configuration options.")
                sys.exit(1)

            gameconfig = {
                "seed":     0,
                "headless": 1,
                "diablo-bin-path":  diablo_bin_path,
                "diablo-data-path": diablo_data_path,
            }
            game = diablo_state.DiabloGame.run(gameconfig)

        # Run the curses application
        curses.wrapper(lambda stdscr: run_tui(stdscr, game))
    else:
        print("Not supported yet")

if __name__ == "__main__":
    main()
