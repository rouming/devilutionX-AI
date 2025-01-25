#!/usr/bin/python3

import mmap
import ctypes
import curses
import time
import sys
import numpy as np
from pynput import keyboard
import enum
import gc
import copy

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
    elif k == keyboard.KeyCode.from_char('z'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
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

class DungeonFlag(enum.Enum):
    Missile               = 1 << 0
    Visible               = 1 << 1
    DeadPlayer            = 1 << 2
    Populated             = 1 << 3
    MissileFireWall       = 1 << 4
    MissileLightningWall  = 1 << 5
    Lit                   = 1 << 6
    Explored              = 1 << 7


class TileProperties(enum.Enum):
    NoneTile         = 0
    Solid            = 1 << 0
    BlockLight       = 1 << 1
    BlockMissile     = 1 << 2
    Transparent      = 1 << 3
    TransparentLeft  = 1 << 4
    TransparentRight = 1 << 5
    Trap             = 1 << 7

class DoorState(enum.Enum):
    DOOR_CLOSED   = 0
    DOOR_OPEN     = 1,
    DOOR_BLOCKED  = 2

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

ObjectTypeStr = [
    "OBJ_L1LIGHT", "OBJ_L1LDOOR", "OBJ_L1RDOOR", "OBJ_SKFIRE",
    "OBJ_LEVER", "OBJ_CHEST1", "OBJ_CHEST2", "OBJ_CHEST3",
    "OBJ_CANDLE1", "OBJ_CANDLE2", "OBJ_CANDLEO", "OBJ_BANNERL",
    "OBJ_BANNERM", "OBJ_BANNERR", "OBJ_SKPILE", "OBJ_SKSTICK1",
    "OBJ_SKSTICK2", "OBJ_SKSTICK3", "OBJ_SKSTICK4", "OBJ_SKSTICK5",
    "OBJ_CRUX1", "OBJ_CRUX2", "OBJ_CRUX3", "OBJ_STAND", "OBJ_ANGEL",
    "OBJ_BOOK2L", "OBJ_BCROSS", "OBJ_NUDEW2R", "OBJ_SWITCHSKL",
    "OBJ_TNUDEM1", "OBJ_TNUDEM2", "OBJ_TNUDEM3", "OBJ_TNUDEM4",
    "OBJ_TNUDEW1", "OBJ_TNUDEW2", "OBJ_TNUDEW3", "OBJ_TORTURE1",
    "OBJ_TORTURE2", "OBJ_TORTURE3", "OBJ_TORTURE4", "OBJ_TORTURE5",
    "OBJ_BOOK2R", "OBJ_L2LDOOR", "OBJ_L2RDOOR", "OBJ_TORCHL",
    "OBJ_TORCHR", "OBJ_TORCHL2", "OBJ_TORCHR2", "OBJ_SARC",
    "OBJ_FLAMEHOLE", "OBJ_FLAMELVR", "OBJ_WATER", "OBJ_BOOKLVR",
    "OBJ_TRAPL", "OBJ_TRAPR", "OBJ_BOOKSHELF", "OBJ_WEAPRACK",
    "OBJ_BARREL", "OBJ_BARRELEX", "OBJ_SHRINEL", "OBJ_SHRINER",
    "OBJ_SKELBOOK", "OBJ_BOOKCASEL", "OBJ_BOOKCASER", "OBJ_BOOKSTAND",
    "OBJ_BOOKCANDLE", "OBJ_BLOODFTN", "OBJ_DECAP", "OBJ_TCHEST1",
    "OBJ_TCHEST2", "OBJ_TCHEST3", "OBJ_BLINDBOOK", "OBJ_BLOODBOOK",
    "OBJ_PEDESTAL", "OBJ_L3LDOOR", "OBJ_L3RDOOR", "OBJ_PURIFYINGFTN",
    "OBJ_ARMORSTAND", "OBJ_ARMORSTANDN", "OBJ_GOATSHRINE",
    "OBJ_CAULDRON", "OBJ_MURKYFTN", "OBJ_TEARFTN", "OBJ_ALTBOY",
    "OBJ_MCIRCLE1", "OBJ_MCIRCLE2", "OBJ_STORYBOOK",
    "OBJ_STORYCANDLE", "OBJ_STEELTOME", "OBJ_WARARMOR", "OBJ_WARWEAP",
    "OBJ_TBCROSS", "OBJ_WEAPONRACK", "OBJ_WEAPONRACKN",
    "OBJ_MUSHPATCH", "OBJ_LAZSTAND", "OBJ_SLAINHERO", "OBJ_SIGNCHEST",
    "OBJ_BOOKSHELFR", "OBJ_POD", "OBJ_PODEX", "OBJ_URN", "OBJ_URNEX",
    "OBJ_L5BOOKS", "OBJ_L5CANDLE", "OBJ_L5LDOOR", "OBJ_L5RDOOR",
    "OBJ_L5LEVER", "OBJ_L5SARC", "OBJ_LAST"
]

# Create enum from strings
ObjectType = enum.Enum('ObjectType',
                       [(s, i) for i, s in enumerate(ObjectTypeStr)])

class Object(ctypes.Structure):
    _fields_ = [
        ("_otype", ctypes.c_byte),
        ("applyLighting", ctypes.c_byte),
        ("_oTrapFlag", ctypes.c_byte),
        ("_oDoorFlag", ctypes.c_byte),
	("position", ctypes.c_int),
	("_oAnimFlag", ctypes.c_uint),
	("_oAnimData", ctypes.c_longlong),
	("_oAnimDelay", ctypes.c_int),
	("_oAnimCnt", ctypes.c_int),
	("_oAnimLen", ctypes.c_uint),
	("_oAnimFrame", ctypes.c_uint),
        ("_oAnimWidth", ctypes.c_ushort),
	("_oDelFlag", ctypes.c_byte),
	("_oBreak", ctypes.c_byte),
	("_oSolidFlag", ctypes.c_byte),
	("_oMissFlag", ctypes.c_byte),
	("selectionRegion", ctypes.c_byte),
	("_oPreFlag", ctypes.c_byte),
	("_olid", ctypes.c_int),
	("_oRndSeed", ctypes.c_uint),
        ("_oVar1", ctypes.c_int),
	("_oVar2", ctypes.c_int),
	("_oVar3", ctypes.c_int),
	("_oVar4", ctypes.c_int),
	("_oVar5", ctypes.c_int),
	("_oVar6", ctypes.c_uint),
	("_oVar8", ctypes.c_int),
	("bookMessage", ctypes.c_short),
    ]

class DiabloSharedHeader(ctypes.Structure):
    _fields_ = [
        ("maxdun" ,      (ctypes.c_short * 2)),
        ("dmax",         (ctypes.c_short * 2)),
        ("max_monsters", ctypes.c_ushort),
        ("max_objects",  ctypes.c_ushort),
        ("max_tiles"  ,  ctypes.c_ushort),
        ("num_mtypes",   ctypes.c_ushort),
    ]

def to_object(d, pos):
    return d.Objects[abs(d.dObject_np[pos]) - 1]

def is_door_closed(obj):
    return obj._oVar4 == DoorState.DOOR_CLOSED.value

def is_barrel(obj):
    return obj._otype in (ObjectType.OBJ_BARREL.value,
                          ObjectType.OBJ_BARRELEX.value,
                          ObjectType.OBJ_POD.value,
                          ObjectType.OBJ_PODEX.value,
                          ObjectType.OBJ_URN.value,
                          ObjectType.OBJ_URNEX.value)

def is_chest(obj):
    return obj._otype in (ObjectType.OBJ_CHEST1.value,
                          ObjectType.OBJ_CHEST2.value,
                          ObjectType.OBJ_CHEST3.value,
                          ObjectType.OBJ_TCHEST1.value,
                          ObjectType.OBJ_TCHEST2.value,
                          ObjectType.OBJ_TCHEST3.value,
                          ObjectType.OBJ_SIGNCHEST.value)

def is_sarcophagus(obj):
    return obj._otype in (ObjectType.OBJ_SARC.value,
                          ObjectType.OBJ_L5SARC.value)

def is_floor(d, pos):
    return not (d.SOLData[d.dPiece_np[pos]] & \
                (TileProperties.Solid.value | TileProperties.BlockMissile.value))

def is_arch(d, pos):
    return d.dSpecial_np[pos] > 0

def is_wall(d, pos):
    return not is_floor(d, pos) and not is_arch(d, pos)

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
            ("Objects",               (Object * hdr.max_objects)),
            ("dItem",        (ctypes.c_ubyte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dTransVal",    (ctypes.c_ubyte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dFlags",       (ctypes.c_ubyte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dPlayer",      (ctypes.c_ubyte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dMonster",     (ctypes.c_short * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dCorpse",      (ctypes.c_ubyte * hdr.maxdun[0]) * hdr.maxdun[1]),
            # Contains the object numbers (objects array indices) of the map.
            # Large objects have negative id for their extended area.
            ("dObject",        (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dPiece",         (ctypes.c_ushort * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dSpecial",       (ctypes.c_byte * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("AutomapView",    (ctypes.c_ubyte * hdr.dmax[0]) * hdr.dmax[1]),
            # TileProperties
            ("SOLData",        (ctypes.c_ubyte * hdr.max_tiles)),
        ]

    diablo = DiabloShared.from_buffer(buf)

    # Create numpy arrays instead of regular
    for field_name, field_type in diablo._fields_:
        field_value = getattr(diablo, field_name)
        if not isinstance(field_value, ctypes.Array):
            continue

        np_view = np.ctypeslib.as_array(field_value)
        setattr(diablo, field_name + "_np", np_view)

    return diablo

class MapRect:
    # Left top
    lt  = None
    # Right bottom
    rb = None

    width = 0
    height = 0

def get_radius(d, stdscr):
    height, width = stdscr.getmaxyx()
    width = min(width, d.maxdun[0])
    height = min(height, d.maxdun[1])
    return (width // 2, height // 2 - 3)

def get_map_rect(d, radius):
    x_min = max(d.player.position.tile.x - radius[0], 0)
    x_max = min(d.player.position.tile.x + radius[0], d.maxdun[0])
    y_min = max(d.player.position.tile.y - radius[1], 0)
    y_max = min(d.player.position.tile.y + radius[1], d.maxdun[1])

    map_rect = MapRect()
    map_rect.lt = np.array([x_min, y_min])
    map_rect.rb = np.array([x_max, y_max])

    map_rect.width  = map_rect.rb[0] - map_rect.lt[0]
    map_rect.height = map_rect.rb[1] - map_rect.lt[1]

    return map_rect


def display_matrix(stdscr, m):
    rows, cols = m.shape

    # Get the screen size
    height, width = stdscr.getmaxyx()

    y = height // 2 - rows // 2
    x = width // 2 - cols // 2

    for row in range(rows):
        for col in range(cols):
            s = m[row, col]
            if len(s) == 0:
                s = '-'
            stdscr.addstr(row + y, col + x, "%s" % s)

def display_diablo_state(stdscr, d_shared):

    # Unfortunately (performance-wise) we have to make a deep copy to
    # prevent partial or complete changes of the state in the middle
    # of this routine
    d = copy.deepcopy(d_shared)

    # Get the screen size
    height, width = stdscr.getmaxyx()

    hdr_msg = "Diablo ticks: %d; Position: %02d,%02d; Kills: %003d" % \
        (d.game_tick,
         d.player.position.tile.x,
         d.player.position.tile.y,
         np.sum(d.MonsterKillCounts_np))
    quit_msg = "Press 'ESC' to quit."

    stdscr.addstr(1, width // 2 - len(hdr_msg) // 2, hdr_msg)
    stdscr.addstr(height - 2, width // 2 - len(quit_msg) // 2, quit_msg)

    radius = get_radius(d, stdscr)
    map_rect = get_map_rect(d, radius)

    # Final surroundings matrix. We can reach the end of the map, so
    # not all values of the surroundings matrix can be filled in
    surroundings = np.full((map_rect.height, map_rect.width), '-', dtype=str)

    for i_off in range(map_rect.width):
        i = map_rect.lt[0] + i_off

        for j_off in range(map_rect.height):
            j = map_rect.lt[1] + j_off
            pos = (i, j)

            s = '-'
            if d.dFlags_np[pos] & DungeonFlag.Explored.value:
                if is_wall(d, pos):
                    s = '#'

                obj = to_object(d, pos)
                if obj._oDoorFlag:
                    if is_door_closed(obj):
                        s = 'D'
                    else:
                        s = 'O'

            if d.dFlags_np[pos] & DungeonFlag.Visible.value:
                if d.dFlags_np[pos] & DungeonFlag.Missile.value:
                    s = '%'
                if d.dMonster_np[pos] > 0:
                    s = '@'

                    obj = to_object(d, pos)
                if is_barrel(obj) and obj._oSolidFlag:
                    s = 'B'
                elif is_chest(obj):
                    if obj.selectionRegion != 0:
                        s = 'C'
                    else:
                        s = 'c'
#                elif is_sarcophagus(obj):
#                    if obj.selectionRegion != 0:
#                        s = 'S'
#                    else:
#                        s = 's'

            if pos == (d.player.position.tile.x,
                       d.player.position.tile.y):
                # Player
                s = 'o'

            surroundings[j_off, i_off] = s

    display_matrix(stdscr, surroundings)

def main(stdscr):
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

    # Open the file and map it to memory
    f = open("/tmp/diablo.shared", "r+b")
    mmapped_file = mmap.mmap(f.fileno(), 0)
    diablo = map_DiabloShared(mmapped_file)

    # Main loop
    while running:
        # Clear the screen
        stdscr.clear()

        # Get the screen size
        height, width = stdscr.getmaxyx()

        # Display messages
        message = "Diablo ticks: %d" % 1000
        quit_message = "Press 'ESC' to quit."

        stdscr.addstr(1, width // 2 - len(message) // 2, message)
        stdscr.addstr(height - 2, width // 2 - len(quit_message) // 2, quit_message)

        display_diablo_state(stdscr, diablo)

        new_key = last_key
        if prev_key != new_key:
            prev_key = new_key

            # Get an entry to submit
            entry = diablo.input_queue.get_entry_to_submit()
            if entry:
                entry.type = new_key
                entry.data = 0
                diablo.input_queue.submit()

        # Refresh the screen to show the content
        stdscr.refresh()

        # Add a delay
        time.sleep(0.01)

    # Close the memory map
    # XXX mmapped_file.close()
    f.close()

    # Wait for listener to stop
    keylistener.stop()

# Run the curses application
curses.wrapper(main)
