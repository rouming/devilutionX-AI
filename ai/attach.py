#!/usr/bin/python3

import mmap
import ctypes
import time
import sys
import numpy as np
from pynput import keyboard
import gc

import ring

# Flag to control the main loop
running = True

# Global variable to track the last key pressed
prev_key = 0
last_key = 0

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
    elif k == keyboard.KeyCode.from_char('z'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == keyboard.Key.esc and press:
        print("Escape key pressed. Exiting...")
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

class Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_byte),
        ("y", ctypes.c_byte)
    ]

class ActorPosition(ctypes.Structure):
    _fields_ = [
        ("tile", Point),
        ("future", Point),
        ("last", Point),
        ("old", Point),
        ("temp", Point)
    ]

class PlayerState(ctypes.Structure):
    _fields_ = [
        ("lightId", ctypes.c_int),
        ("_pNumInv", ctypes.c_int),
        ("_pStrength", ctypes.c_int),
        ("_pBaseStr", ctypes.c_int),
        ("_pMagic", ctypes.c_int),
        ("_pBaseMag", ctypes.c_int),
        ("_pDexterity", ctypes.c_int),
        ("_pBaseDex", ctypes.c_int),
        ("_pVitality", ctypes.c_int),
        ("_pBaseVit", ctypes.c_int),
        ("_pStatPts", ctypes.c_int),
        ("_pDamageMod", ctypes.c_int),
        ("_pHPBase", ctypes.c_int),
        ("_pMaxHPBase", ctypes.c_int),
        ("_pHitPoints", ctypes.c_int),
        ("_pMaxHP", ctypes.c_int),
        ("_pHPPer", ctypes.c_int),
        ("_pManaBase", ctypes.c_int),
        ("_pMaxManaBase", ctypes.c_int),
        ("_pMana", ctypes.c_int),
        ("_pMaxMana", ctypes.c_int),
        ("_pManaPer", ctypes.c_int),
        ("_pIMinDam", ctypes.c_int),
        ("_pIMaxDam", ctypes.c_int),
        ("_pIAC", ctypes.c_int),
        ("_pIBonusDam", ctypes.c_int),
        ("_pIBonusToHit", ctypes.c_int),
        ("_pIBonusAC", ctypes.c_int),
        ("_pIBonusDamMod", ctypes.c_int),
        ("_pIGetHit", ctypes.c_int),
        ("_pIEnAc", ctypes.c_int),
        ("_pIFMinDam", ctypes.c_int),
        ("_pIFMaxDam", ctypes.c_int),
        ("_pILMinDam", ctypes.c_int),
        ("_pILMaxDam", ctypes.c_int),
	("_pExperience", ctypes.c_uint),
        ("_pmode", ctypes.c_byte),
        ("padding1", (ctypes.c_byte * 3)),
        ("position", ActorPosition),
        ("padding2", (ctypes.c_byte * 2)),
    ]

class Monster(ctypes.Structure):
    _fields_ = [
        ("opaque1", (ctypes.c_char * 104)),
    ]

class DiabloSharedHeader(ctypes.Structure):
    _fields_ = [
        ("maxdun" ,      (ctypes.c_short * 2)),
        ("dmax",         (ctypes.c_short * 2)),
        ("max_monsters", ctypes.c_uint),
        ("num_mtypes",   ctypes.c_uint),
    ]

def map_DiabloShared(buf):
    hdr = DiabloSharedHeader.from_buffer(buf)

    class DiabloShared(DiabloSharedHeader):
        _fields_ = [
            ("input_queue",  ring.RingQueue),
            ("events_queue", ring.RingQueue),
            ("player",       PlayerState),
            ("game_tick",    ctypes.c_ulonglong),
            ("LevelMonsterTypeCount", ctypes.c_size_t),
            ("ActiveMonsterCount",    ctypes.c_size_t),
            ("Monsters",              (Monster * hdr.max_monsters)),
            ("ActiveMonsters",        (ctypes.c_uint * hdr.max_monsters)),
            ("MonsterKillCounts",     (ctypes.c_int * hdr.num_mtypes)),
            ("dItem",        (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dTransVal",    (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dFlags",       (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dPlayer",      (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dMonster",     (ctypes.c_short * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dCorpse",      (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dObject",      (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("AutomapView",  (ctypes.c_byte * hdr.dmax[0]) * hdr.dmax[1]),
        ]

    diablo = DiabloShared.from_buffer(buf)

    for field_name, field_type in diablo._fields_:
        field_value = getattr(diablo, field_name)
        if not isinstance(field_value, ctypes.Array):
            continue

        #
        # Careful when building np.array directly from a buffer
        # (no copy) - numpy can fire runtime error during a call
        # ('notzero' for example):
        # "RuntimeError: number of non-zero array elements changed
        #  during function execution."
        #
        NP_COPY = True

        if NP_COPY:
            np_view = np.ctypeslib.as_array(field_value)
        else:
            elem_type = field_type
            np_shape = []
            while issubclass(elem_type, ctypes.Array):
                np_shape.append(elem_type._length_)
                elem_type = elem_type._type_

            if ctypes.sizeof(elem_type) == 1:
                np_type = np.int8
            elif ctypes.sizeof(elem_type) == 2:
                np_type = np.int16
            elif ctypes.sizeof(elem_type) == 4:
                np_type = np.int64
            else:
                assert False, "Unknown type size: %d" % ctypes.sizeof(elem_type)

            field_address = ctypes.addressof(diablo) + \
                getattr(DiabloShared, field_name).offset
            field_size = ctypes.sizeof(field_type)

            np_view = np.frombuffer(field_type.from_address(field_address),
                                    dtype=np_type)
            if len(np_shape) > 1:
                np_view = np_view.reshape(np_shape)

        # Set numpy view
        setattr(diablo, field_name + "_np", np_view)

    return diablo

def print_diablo_state(diablo):
    ActiveMonsters_ind = np.nonzero(diablo.ActiveMonsters_np)
    MonsterKillCounts_ind = np.nonzero(diablo.MonsterKillCounts_np)

    dPlayer_ind = np.nonzero(diablo.dPlayer_np)
    dMonster_ind = np.nonzero(diablo.dMonster_np)
    dFlags_ind = np.nonzero(diablo.dFlags_np)
    dItem_ind = np.nonzero(diablo.dItem_np)

    print("Tick: %d" % diablo.game_tick)
    for i in ActiveMonsters_ind:
        print(f"ActiveMonsters: ({i}): {diablo.ActiveMonsters_np[i]}")
    for i in MonsterKillCounts_ind:
        print(f"MonsterKillCounts: ({i}): {diablo.MonsterKillCounts_np[i]}")
    for i, j in zip(*dPlayer_ind):
        print(f"dPlayer: ({i}, {j}): {diablo.dPlayer_np[i][j]}")
    for i, j in zip(*dItem_ind):
        print(f"dItem: ({i}, {j}): {diablo.dItem_np[i][j]}")
#   for i, j in zip(*dFlags_ind):
#       print(f"dFlags: ({i}, {j}): {diablo.dFlags_np[i][j]}")
#   for i, j in zip(*dMonster_ind):
#      print(f"dMonster: ({i}, {j}): {diablo.dMonster_np[i][j]}")

    print("Player:")
    for field_name, field_type in diablo.player._fields_:
        if not field_name.startswith("opaque"):
            continue
        print("   %s:\t\t%d" % (field_name, getattr(diablo.player, field_name)))



# Start the keyboard listener in non-blocking mode
keylistener = keyboard.Listener(on_press=on_key_press,
                                on_release=on_key_release)
keylistener.start()

# Open the file and map it to memory
with open("/tmp/diablo.shared", "r+b") as f:
    mmapped_file = mmap.mmap(f.fileno(), 0)

    diablo = map_DiabloShared(mmapped_file)

    # Main loop
    while running:
#        print_diablo_state(diablo)

        new_key = last_key
        if prev_key != new_key:
            prev_key = new_key

            # Get an entry to submit
            entry = diablo.input_queue.get_entry_to_submit()
            if entry:
                entry.type = new_key
                entry.data = 0
                diablo.input_queue.submit()

                print(">>> SUBMITTED: {:08b}, nr_events=%d".format(new_key) % \
                      diablo.input_queue.nr_entries_to_submit())



        # Add a delay
        time.sleep(0.01)

    # Close the memory map
    # XXX mmapped_file.close()

# Wait for listener to stop
keylistener.stop()
