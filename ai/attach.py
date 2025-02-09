#!/usr/bin/env python3

import os
import mmap
import ctypes
import curses
import time
import sys
from pynput import keyboard
import enum
import gc
import copy
import tempfile
import subprocess

import ring

import numpy as np
import gymnasium as gym

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

class AnimationInfo(ctypes.Structure):
    _fields_ = [
        ("sprites", ctypes.c_ulonglong),  # OptionalClxSpriteList
        ("ticksPerFrame", ctypes.c_int8),
        ("tickCounterOfCurrentFrame", ctypes.c_int8),
        ("numberOfFrames", ctypes.c_int8),
        ("currentFrame", ctypes.c_int8),
        ("isPetrified", ctypes.c_bool),
        ("relevantFramesForDistributing_", ctypes.c_int8),
        ("skippedFramesFromPreviousAnimation_", ctypes.c_int8),
        ("tickModifier_", ctypes.c_uint16),
        ("ticksSinceSequenceStarted_", ctypes.c_int16),
    ]

class WorldTilePosition(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int8),
        ("y", ctypes.c_int8)
    ]

class ActorPosition(ctypes.Structure):
    _fields_ = [
        ("tile", WorldTilePosition),
        ("future", WorldTilePosition),
        ("last", WorldTilePosition),
        ("old", WorldTilePosition),
        ("temp", WorldTilePosition)
    ]

class interface_mode(enum.Enum):
    WM_DIABNEXTLVL = enum.auto()
    WM_DIABPREVLVL = enum.auto()
    WM_DIABRTNLVL = enum.auto()
    WM_DIABSETLVL = enum.auto()
    WM_DIABWARPLVL = enum.auto()
    WM_DIABTOWNWARP = enum.auto()
    WM_DIABTWARPUP = enum.auto()
    WM_DIABRETOWN = enum.auto()
    WM_DIABNEWGAME = enum.auto()
    WM_DIABLOADGAME = enum.auto()

    # Asynchronous loading events.
    WM_PROGRESS = enum.auto()
    WM_ERROR = enum.auto()
    WM_DONE = enum.auto()

    WM_FIRST = WM_DIABNEXTLVL
    WM_LAST = WM_DONE

class TriggerStruct(ctypes.Structure):
    _fields_ = [
	("position", WorldTilePosition),
        ("_tmsg", ctypes.c_uint8), # interface_mode
        ("_tlvl", ctypes.c_int32)
    ]

class PLR_MODE(enum.Enum):
    PM_STAND           = 0
    PM_WALK_NORTHWARDS = 1
    PM_WALK_SOUTHWARDS = 2
    PM_WALK_SIDEWAYS   = 3
    PM_ATTACK          = 4
    PM_RATTACK         = 5
    PM_BLOCK           = 6
    PM_GOTHIT          = 7
    PM_DEATH           = 8
    PM_SPELL           = 9
    PM_NEWLVL          = 10
    PM_QUIT            = 11

class action_id(enum.Enum):
    ACTION_WALK        = -2 # Automatic walk when using gamepad
    ACTION_NONE        = -1
    ACTION_ATTACK      = 9
    ACTION_RATTACK     = 10
    ACTION_SPELL       = 12
    ACTION_OPERATE     = 13
    ACTION_DISARM      = 14
    ACTION_PICKUPITEM  = 15 # put item in hand (inventory screen open)
    ACTION_PICKUPAITEM = 16 # put item in inventory
    ACTION_TALK        = 17
    ACTION_OPERATETK   = 18 # operate via telekinesis
    ACTION_ATTACKMON   = 20
    ACTION_ATTACKPLR   = 21
    ACTION_RATTACKMON  = 22
    ACTION_RATTACKPLR  = 23
    ACTION_SPELLMON    = 24
    ACTION_SPELLPLR    = 25
    ACTION_SPELLWALL   = 26

class PlayerState(ctypes.Structure):
    _fields_ = [
        ("lightId", ctypes.c_int32),
        ("_pNumInv", ctypes.c_int32),
        ("_pStrength", ctypes.c_int32),
        ("_pBaseStr", ctypes.c_int32),
        ("_pMagic", ctypes.c_int32),
        ("_pBaseMag", ctypes.c_int32),
        ("_pDexterity", ctypes.c_int32),
        ("_pBaseDex", ctypes.c_int32),
        ("_pVitality", ctypes.c_int32),
        ("_pBaseVit", ctypes.c_int32),
        ("_pStatPts", ctypes.c_int32),
        ("_pDamageMod", ctypes.c_int32),
        ("_pHPBase", ctypes.c_int32),
        ("_pMaxHPBase", ctypes.c_int32),
        ("_pHitPoints", ctypes.c_int32),
        ("_pMaxHP", ctypes.c_int32),
        ("_pHPPer", ctypes.c_int32),
        ("_pManaBase", ctypes.c_int32),
        ("_pMaxManaBase", ctypes.c_int32),
        ("_pMana", ctypes.c_int32),
        ("_pMaxMana", ctypes.c_int32),
        ("_pManaPer", ctypes.c_int32),
        ("_pIMinDam", ctypes.c_int32),
        ("_pIMaxDam", ctypes.c_int32),
        ("_pIAC", ctypes.c_int32),
        ("_pIBonusDam", ctypes.c_int32),
        ("_pIBonusToHit", ctypes.c_int32),
        ("_pIBonusAC", ctypes.c_int32),
        ("_pIBonusDamMod", ctypes.c_int32),
        ("_pIGetHit", ctypes.c_int32),
        ("_pIEnAc", ctypes.c_int32),
        ("_pIFMinDam", ctypes.c_int32),
        ("_pIFMaxDam", ctypes.c_int32),
        ("_pILMinDam", ctypes.c_int32),
        ("_pILMaxDam", ctypes.c_int32),
        ("_pExperience", ctypes.c_uint32),
        ("_pmode", ctypes.c_uint8),
        ("destAction", ctypes.c_int8), # action_id
        ("plractive", ctypes.c_bool),
        ("padding1", (ctypes.c_int8 * 1)),
        ("position", ActorPosition),
        ("padding2", (ctypes.c_int8 * 2)),
        ("AnimInfo", AnimationInfo),
    ]

class Monster(ctypes.Structure):
    _fields_ = [
        ("uniqueMonsterTRN", ctypes.c_ulonglong),  # std::unique_ptr<uint8_t[]> equivalent
        ("animInfo", AnimationInfo),
        ("maxHitPoints", ctypes.c_int),
        ("hitPoints", ctypes.c_int),
        ("flags", ctypes.c_uint32),
        ("rndItemSeed", ctypes.c_uint32),
        ("aiSeed", ctypes.c_uint32),
        ("golemToHit", ctypes.c_uint16),
        ("resistance", ctypes.c_uint16),
        ("talkMsg", ctypes.c_int16),  # _speech_id
        ("goalVar1", ctypes.c_int16),
        ("goalVar2", ctypes.c_int8),
        ("goalVar3", ctypes.c_int8),
        ("var1", ctypes.c_int16),
        ("var2", ctypes.c_int16),
        ("var3", ctypes.c_int8),
        ("position", ActorPosition),
        ("goal", ctypes.c_int8),  # MonsterGoal
        ("enemyPosition", WorldTilePosition),
        ("levelType", ctypes.c_uint8),
        ("mode", ctypes.c_int8),  # MonsterMode
        ("pathCount", ctypes.c_uint8),
        ("direction", ctypes.c_int8),  # Direction
        ("enemy", ctypes.c_uint8),
        ("isInvalid", ctypes.c_bool),
        ("ai", ctypes.c_int8),  # MonsterAIID
        ("intelligence", ctypes.c_uint8),
        ("activeForTicks", ctypes.c_uint8),
        ("uniqueType", ctypes.c_int8),  # UniqueMonsterType
        ("uniqTrans", ctypes.c_uint8),
        ("corpseId", ctypes.c_int8),
        ("whoHit", ctypes.c_int8),
        ("minDamage", ctypes.c_uint8),
        ("maxDamage", ctypes.c_uint8),
        ("minDamageSpecial", ctypes.c_uint8),
        ("maxDamageSpecial", ctypes.c_uint8),
        ("armorClass", ctypes.c_uint8),
        ("leader", ctypes.c_uint8),
        ("leaderRelation", ctypes.c_int8),  # LeaderRelation
        ("packSize", ctypes.c_uint8),
        ("lightId", ctypes.c_int8),
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
        ("_otype", ctypes.c_int8),
        ("applyLighting", ctypes.c_int8),
        ("_oTrapFlag", ctypes.c_int8),
        ("_oDoorFlag", ctypes.c_int8),
        ("position", ctypes.c_int32),
        ("_oAnimFlag", ctypes.c_uint32),
        ("_oAnimData", ctypes.c_int64),
        ("_oAnimDelay", ctypes.c_int32),
        ("_oAnimCnt", ctypes.c_int32),
        ("_oAnimLen", ctypes.c_uint32),
        ("_oAnimFrame", ctypes.c_uint32),
        ("_oAnimWidth", ctypes.c_uint16),
        ("_oDelFlag", ctypes.c_int8),
        ("_oBreak", ctypes.c_int8),
        ("_oSolidFlag", ctypes.c_int8),
        ("_oMissFlag", ctypes.c_int8),
        ("selectionRegion", ctypes.c_int8),
        ("_oPreFlag", ctypes.c_int8),
        ("_olid", ctypes.c_int32),
        ("_oRndSeed", ctypes.c_uint32),
        ("_oVar1", ctypes.c_int32),
        ("_oVar2", ctypes.c_int32),
        ("_oVar3", ctypes.c_int32),
        ("_oVar4", ctypes.c_int32),
        ("_oVar5", ctypes.c_int32),
        ("_oVar6", ctypes.c_uint32),
        ("_oVar8", ctypes.c_int32),
        ("bookMessage", ctypes.c_int16),
    ]

class Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32)
    ]

class Item(ctypes.Structure):
    _fields_ = [
        ("_iSeed", ctypes.c_uint32),
        ("_iCreateInfo", ctypes.c_uint16),
        ("_itype", ctypes.c_int8),
        ("_iAnimFlag", ctypes.c_uint8),
        ("position", Point),
        ("AnimInfo", AnimationInfo),
        ("_iDelFlag", ctypes.c_uint8),
        ("selectionRegion", ctypes.c_int8),
        ("_iPostDraw", ctypes.c_uint8),
        ("_iIdentified", ctypes.c_uint8),
        ("_iMagical", ctypes.c_uint8),
        ("_iName", ctypes.c_char * 64),
        ("_iIName", ctypes.c_char * 64),
        ("_iLoc", ctypes.c_uint8),
        ("_iClass", ctypes.c_uint8),
        ("_iCurs", ctypes.c_uint8),
        ("_ivalue", ctypes.c_int32),
        ("_iIvalue", ctypes.c_int32),
        ("_iMinDam", ctypes.c_uint8),
        ("_iMaxDam", ctypes.c_uint8),
        ("_iAC", ctypes.c_int16),
        ("_iFlags", ctypes.c_uint32),
        ("_iMiscId", ctypes.c_int8),
        ("_iSpell", ctypes.c_int8),
        ("IDidx", ctypes.c_int16),
        ("_iCharges", ctypes.c_int32),
        ("_iMaxCharges", ctypes.c_int32),
        ("_iDurability", ctypes.c_int32),
        ("_iMaxDur", ctypes.c_int32),
        ("_iPLDam", ctypes.c_int16),
        ("_iPLToHit", ctypes.c_int16),
        ("_iPLAC", ctypes.c_int16),
        ("_iPLStr", ctypes.c_int16),
        ("_iPLMag", ctypes.c_int16),
        ("_iPLDex", ctypes.c_int16),
        ("_iPLVit", ctypes.c_int16),
        ("_iPLFR", ctypes.c_int16),
        ("_iPLLR", ctypes.c_int16),
        ("_iPLMR", ctypes.c_int16),
        ("_iPLMana", ctypes.c_int16),
        ("_iPLHP", ctypes.c_int16),
        ("_iPLDamMod", ctypes.c_int16),
        ("_iPLGetHit", ctypes.c_int16),
        ("_iPLLight", ctypes.c_int16),
        ("_iSplLvlAdd", ctypes.c_int8),
        ("_iRequest", ctypes.c_uint8),
        ("_iUid", ctypes.c_int32),
        ("_iFMinDam", ctypes.c_int16),
        ("_iFMaxDam", ctypes.c_int16),
        ("_iLMinDam", ctypes.c_int16),
        ("_iLMaxDam", ctypes.c_int16),
        ("_iPLEnAc", ctypes.c_int16),
        ("_iPrePower", ctypes.c_int8),
        ("_iSufPower", ctypes.c_int8),
        ("_iVAdd1", ctypes.c_int32),
        ("_iVMult1", ctypes.c_int32),
        ("_iVAdd2", ctypes.c_int32),
        ("_iVMult2", ctypes.c_int32),
        ("_iMinStr", ctypes.c_int8),
        ("_iMinMag", ctypes.c_uint8),
        ("_iMinDex", ctypes.c_int8),
        ("_iStatFlag", ctypes.c_uint8),
        ("_iDamAcFlags", ctypes.c_uint8),
        ("dwBuff", ctypes.c_uint32),
    ]


class DiabloSharedHeader(ctypes.Structure):
    _fields_ = [
        ("maxdun" ,      (ctypes.c_int16 * 2)),
        ("dmax",         (ctypes.c_int16 * 2)),
        ("max_monsters", ctypes.c_uint32),
        ("max_objects",  ctypes.c_uint32),
        ("max_tiles"  ,  ctypes.c_uint32),
        ("max_items"  ,  ctypes.c_uint32),
        ("num_mtypes",   ctypes.c_uint32),
        ("max_triggers", ctypes.c_uint32),
    ]

def to_object(d, pos):
    obj_id = d.dObject_np[pos]
    if obj_id != 0:
        return d.Objects[abs(obj_id) - 1]
    return None

def is_door_closed(obj):
    return obj._oVar4 == DoorState.DOOR_CLOSED.value

def is_door(obj):
    return obj._oDoorFlag

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

def is_trigger(d, pos):
    for trig in d.trigs[:d.numtrigs]:
        if trig.position.x == pos[0] and trig.position.y == pos[1]:
            return True
    return False

def map_DiabloShared(buf):
    hdr = DiabloSharedHeader.from_buffer(buf)

    class DiabloShared(DiabloSharedHeader):
        _fields_ = [
            ("input_queue",  ring.RingQueue),
            ("events_queue", ring.RingQueue),
            ("player",       PlayerState),
            ("game_tick",    ctypes.c_ulonglong),

            #
            # Monsters
            #
            ("LevelMonsterTypeCount", ctypes.c_size_t),
            ("ActiveMonsterCount",    ctypes.c_size_t),
            ("Monsters",              (Monster * hdr.max_monsters)),
            ("ActiveMonsters",        (ctypes.c_uint * hdr.max_monsters)),
            ("MonsterKillCounts",     (ctypes.c_int * hdr.num_mtypes)),

            #
            # Objects
            #
            ("Objects",               (Object * hdr.max_objects)),
            ("AvailableObjects",       ctypes.c_int32 * hdr.max_objects),
            ("ActiveObjects",          ctypes.c_int32 * hdr.max_objects),
            ("ActiveObjectCount",      ctypes.c_int32),
            ("padding1",               ctypes.c_uint32),

            #
            # Items
            #
            ("Items",           (Item * (hdr.max_items + 1))),
            ("ActiveItems",     (ctypes.c_uint8 * hdr.max_items)),
            ("ActiveItemCount", ctypes.c_uint8),
            ("dItem",           (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),

            #
            # Gendung
            #
            ("dTransVal",    (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dFlags",       (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dPlayer",      (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dMonster",     (ctypes.c_int16 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dCorpse",      (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            # Contains the object numbers (objects array indices) of the map.
            # Large objects have negative id for their extended area.
            ("dObject",        (ctypes.c_int8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dPiece",         (ctypes.c_uint16 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dSpecial",       (ctypes.c_int8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            # TileProperties
            ("SOLData",        (ctypes.c_uint8 * hdr.max_tiles)),

            #
            # Trigs
            #
            ("padding2",      ctypes.c_uint8),
            ("numtrigs",      ctypes.c_int32),
            ("trigs",         TriggerStruct * hdr.max_triggers),
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
    x_min = max(d.player.position.future.x - radius[0], 0)
    x_max = min(d.player.position.future.x + radius[0], d.maxdun[0])
    y_min = max(d.player.position.future.y - radius[1], 0)
    y_max = min(d.player.position.future.y + radius[1], d.maxdun[1])

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

    class DungeonFlag(enum.Enum):
        Player         = 1<<0
        Wall           = 1<<1
        Trigger        = 1<<2
        DoorOpened     = 1<<3
        DoorClosed     = 1<<4
        Missile        = 1<<5
        Monster        = 1<<6
        Barrel         = 1<<7
        ChestOpened    = 1<<8
        ChestClosed    = 1<<9
        SarcophOpened  = 1<<10
        SarcophClosed  = 1<<11
        Item           = 1<<12

    def __init__(self, env_config):
        self.config = env_config
        self.seed = env_config["seed"] ^ env_config.worker_index

        cfg_file = open("diablo.ini.tmpl", "r")
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
        entry = diablo.input_queue.get_entry_to_submit()
        assert entry
        entry.type = \
            ring.RingEntryType.RING_ENTRY_KEY_SAVE |
            ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        entry.data = 0
        diablo.input_queue.submit()

        # Busy-loop for actual key acceptance
        while diablo.input_queue.nr_entries_to_submit() != \
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

    def get_environment(self, d):
        # Full dungeon
        dun_rect = get_map_rect(d, d.maxdun)
        env = np.zeros((dun_rect.height, dun_rect.width), dtype=np.uint16)

        for i_off in range(map_rect.width):
            i = map_rect.lt[0] + i_off

            for j_off in range(map_rect.height):
                j = map_rect.lt[1] + j_off
                pos = (i, j)
                obj = to_object(d, pos)

                s = 0
                if d.dFlags_np[pos] & DungeonFlag.Explored.value:
                    if is_wall(d, pos):
                        s |= DungeonFlag.Wall
                    if is_trigger(d, pos):
                        s |= DungeonFlag.Trigger
                    if obj is not None and is_door(obj):
                        if is_door_closed(obj):
                            s |= DungeonFlag.DoorClosed
                        else:
                            s |= DungeonFlag.DoorOpened

                if d.dFlags_np[pos] & DungeonFlag.Lit.value:
                    if d.dFlags_np[pos] & DungeonFlag.Missile.value:
                        s |= DungeonFlag.Missile
                    if d.dMonster_np[pos] > 0:
                        s |= DungeonFlag.Monster

                    if obj is not None:
                        if is_barrel(obj) and obj._oSolidFlag:
                            s |= DungeonFlag.Barrel
                        elif is_chest(obj):
                            if obj.selectionRegion != 0:
                                s |= DungeonFlag.ChestClosed
                            else:
                                s |= DungeonFlag.ChestOpened
                        elif is_sarcophagus(obj):
                            if obj.selectionRegion != 0:
                                s |= DungeonFlag.SarcophClosed
                            else:
                                s |= DungeonFlag.SarcophOpened
                    if d.dItem_np[pos] > 0:
                        s |= DungeonFlag.Item

                if pos == (d.player.position.future.x,
                           d.player.position.future.y):
                    # Player
                    s |= DungeonFlag.Player

                env[j_off, i_off] = s

        return env

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
         PLR_MODE(d.player._pmode).name)
    quit_msg = "Press 'ESC' to quit"

    stdscr.addstr(0, width // 2 - len(hdr_msg) // 2, hdr_msg)
    stdscr.addstr(height - 1, width // 2 - len(quit_msg) // 2, quit_msg)

    def interesting_objects(oid):
        obj = d.Objects[oid]
        if is_door(obj):
            return 1 if is_door_closed(obj) else 0
        elif is_barrel(obj):
            return 1 if obj._oSolidFlag else 0
        elif is_chest(obj) or is_sarcophagus(obj):
            return 1 if obj.selectionRegion != 0 else 0
        return 0

    obj_cnt = sum(map(interesting_objects, d.ActiveObjects))
    total_hp = sum(map(lambda mid: d.Monsters[mid].hitPoints, d.ActiveMonsters))

    msg = "Animation: ticksPerFrame %d; tickCntOfFrame %d; frames %d; frame %d" % \
        (d.player.AnimInfo.ticksPerFrame,
         d.player.AnimInfo.tickCounterOfCurrentFrame,
         d.player.AnimInfo.numberOfFrames,
         d.player.AnimInfo.currentFrame)
    stdscr.addstr(1, width // 2 - len(msg) // 2, msg)

    msg = "Total: monsters HP %d, items %d, objects %d" % \
        (total_hp, d.ActiveItemCount, obj_cnt)
    stdscr.addstr(2, width // 2 - len(msg) // 2, msg)

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
            obj = to_object(d, pos)

            s = '-'
            if d.dFlags_np[pos] & DungeonFlag.Explored.value:
                if is_wall(d, pos):
                    s = '#'
                if is_trigger(d, pos):
                    s = '*'
                if obj is not None and is_door(obj):
                    if is_door_closed(obj):
                        s = 'D'
                    else:
                        s = 'O'

            if d.dFlags_np[pos] & DungeonFlag.Lit.value:
                if d.dFlags_np[pos] & DungeonFlag.Missile.value:
                    s = '%'
                if d.dMonster_np[pos] > 0:
                    s = '@'

                if obj is not None:
                    if is_barrel(obj) and obj._oSolidFlag:
                        s = 'B'
                    elif is_chest(obj):
                        if obj.selectionRegion != 0:
                            s = 'C'
                        else:
                            s = 'c'
                    elif is_sarcophagus(obj):
                        if obj.selectionRegion != 0:
                            s = 'S'
                        else:
                            s = 's'
                if d.dItem_np[pos] > 0:
                    s = 'I'

            if pos == (d.player.position.future.x,
                       d.player.position.future.y):
                # Player
                if d.player._pmode == PLR_MODE.PM_DEATH.value:
                    s = 'X'
                else:
                    s = 'o'

            surroundings[j_off, i_off] = s

#    mr = get_map_rect(d, (4,4))
#    vfunc = np.vectorize(lambda o: o[0])
#    display_matrix(stdscr, #vfunc(d.Objects_np[
#        abs(d.dObject_np[mr.lt[0]:mr.rb[0],
#                         mr.lt[1]:mr.rb[1]]).T-1)
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
    path = os.path.abspath("../diablo-states/cfg/shared.mem")
    f = open(path, "r+b")
    mmapped_file = mmap.mmap(f.fileno(), 0)
    diablo = map_DiabloShared(mmapped_file)

    missed_ticks = 0

    # Main loop
    while running:
        # Clear the screen
        stdscr.clear()

        display_diablo_state(stdscr, diablo, missed_ticks)

        new_key = last_key
        if prev_key != new_key:
            prev_key = new_key

            # Get an entry to submit
            entry = diablo.input_queue.get_entry_to_submit()
            if entry:
                entry.type = new_key | \
                    ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
                entry.data = 0
                diablo.input_queue.submit()

        # Refresh the screen to show the content
        stdscr.refresh()

        # Add a delay
        while running and same_ticks(diablo):
            pass

        missed_ticks = update_ticks(diablo)

    # Close the memory map
    # XXX mmapped_file.close()
    f.close()

    # Wait for listener to stop
    keylistener.stop()

# Run the curses application
curses.wrapper(main)
