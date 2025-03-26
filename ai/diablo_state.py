"""
diablo_state.py - Diablo game state related structures and functions
"""

import ctypes
import enum
import mmap
import numpy as np
import os
import subprocess
import tempfile
import time

import ring
import maze

def round_up_int(i, d):
    assert type(i) == int
    assert type(d) == int
    return (i + d - 1) // d * d

class DungeonFlag(enum.Enum):
    Missile               = 1<<0
    Visible               = 1<<1
    DeadPlayer            = 1<<2
    Populated             = 1<<3
    MissileFireWall       = 1<<4
    MissileLightningWall  = 1<<5
    Lit                   = 1<<6
    Explored              = 1<<7

class TileProperties(enum.Enum):
    NoneTile         = 0
    Solid            = 1<<0
    BlockLight       = 1<<1
    BlockMissile     = 1<<2
    Transparent      = 1<<3
    TransparentLeft  = 1<<4
    TransparentRight = 1<<5
    Trap             = 1<<7

class DoorState(enum.Enum):
    DOOR_CLOSED   = 0
    DOOR_OPEN     = 1,
    DOOR_BLOCKED  = 2

class interface_mode(enum.Enum):
    WM_DIABNEXTLVL = 0
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

class Direction(enum.Enum):
    South       = 0
    SouthWest   = enum.auto()
    West        = enum.auto()
    NorthWest   = enum.auto()
    North       = enum.auto()
    NorthEast   = enum.auto()
    East        = enum.auto()
    SouthEast   = enum.auto()
    NoDirection = enum.auto()

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
        # Future tile position. Set at start of walking animation.
        ("future", WorldTilePosition),
        # Tile position of player. Set via network on player input.
        ("last", WorldTilePosition),
        # Most recent position in dPlayer.
        ("old", WorldTilePosition),
        ("temp", WorldTilePosition)
    ]

class TriggerStruct(ctypes.Structure):
    _fields_ = [
	("position", WorldTilePosition),
        ("_tmsg", ctypes.c_uint8), # interface_mode
        ("_tlvl", ctypes.c_int32)
    ]

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
        ("_pNFrames", ctypes.c_int8),
        ("_pWFrames", ctypes.c_int8),
        ("_pAFrames", ctypes.c_int8),
        ("_pAFNum", ctypes.c_int8),
        ("_pSFrames", ctypes.c_int8),
        ("_pSFNum", ctypes.c_int8),
        ("_pHFrames", ctypes.c_int8),
        ("_pDFrames", ctypes.c_int8),
        ("_pBFrames", ctypes.c_int8),
        ("position", ActorPosition),
        ("_pLevel", ctypes.c_uint8), # character level
        ("plrlevel", ctypes.c_uint8), # dungeon level
        ("_pdir", ctypes.c_uint8), # Direction faced by player (direction enum)
        ("padding1", (ctypes.c_int8 * 7)),
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

class DiabloStateHeader(ctypes.Structure):
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

def is_interactable(obj):
    return obj.selectionRegion != 0

def is_breakable(obj):
    return obj._oBreak == 1

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

def is_crucifix(obj):
    return obj._otype in (ObjectType.OBJ_CRUX1.value,
                          ObjectType.OBJ_CRUX2.value,
                          ObjectType.OBJ_CRUX3.value)

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

def is_game_paused(d):
    return d.PauseMode != 0

def is_player_dead(d):
    return d.player._pmode == PLR_MODE.PM_DEATH.value

def player_position(d):
    return (d.player.position.tile.x, d.player.position.tile.y)

def player_direction(d):
    # Compensate dungeon 45CW rotation
    return (d.player._pdir - 1) % (len(Direction) - 1)

def count_active_objects(d):
    def interesting_objects(oid):
        obj = d.Objects[oid]
        if is_barrel(obj):
            return 1 if obj._oSolidFlag else 0
        elif is_chest(obj) or is_sarcophagus(obj) or is_crucifix(obj):
            return 1 if is_interactable(obj) else 0
        return 0
    return sum(map(interesting_objects, d.ActiveObjects))

def get_closed_doors_ids(d):
    closed_doors = []
    for oid in d.ActiveObjects:
        obj = d.Objects[oid]
        if is_door(obj) and is_door_closed(obj):
            closed_doors.append(oid)
    return closed_doors

def count_active_items(d):
    return d.ActiveItemCount

def count_active_monsters(d):
    return d.ActiveMonsterCount

def count_active_monsters_total_hp(d):
    return sum(map(lambda mid: d.Monsters[mid].hitPoints, d.ActiveMonsters))

def count_explored_tiles(d):
    bits = DungeonFlag.Explored.value
    return np.sum((d.dFlags_np & bits) == bits)

def find_trigger(d, tmsg):
    for trig in d.trigs:
        if trig._tmsg == tmsg.value:
            return trig
    return None

class Rect:
    # Top left
    lt = None
    width  = 0
    height = 0

class EnvironmentRect:
    # Source rectangle
    srect = Rect()
    # Destination rectangle
    drect = Rect()

    def __init__(self, d, radius=None):
        if radius is not None:
            assert d

            pos = player_position(d)

            x_min = max(pos[0] - radius[0], 0)
            x_max = min(pos[0] + radius[0], d.maxdun[0])
            y_min = max(pos[1] - radius[1], 0)
            y_max = min(pos[1] + radius[1], d.maxdun[1])

            self.srect.lt     = np.array([x_min, y_min])
            self.srect.width  = x_max - self.srect.lt[0]
            self.srect.height = y_max - self.srect.lt[1]

            # Place player position in the center of a destination rectangle
            self.drect.lt     = radius - (pos - self.srect.lt)
            self.drect.width  = radius[0] * 2
            self.drect.height = radius[1] * 2
        else:
            self.srect.lt     = np.array([0, 0])
            self.srect.width  = d.maxdun[0]
            self.srect.height = d.maxdun[1]
            self.drect        = self.srect

class EnvironmentFlag(enum.Enum):
    Player         = 1<<0
    Wall           = 1<<1
    Trigger        = 1<<2
    DoorOpened     = 1<<3
    DoorClosed     = 1<<4
    Missile        = 1<<5
    Monster        = 1<<6
    UnknownObject  = 1<<7
    Crucifix       = 1<<8
    Barrel         = 1<<9
    Chest          = 1<<10
    Sarcophagus    = 1<<11
    Item           = 1<<12
    Explored       = 1<<13
    Visible        = 1<<14
    Interactable   = 1<<15

def get_environment(d, radius=None, ignore_explored_visible=False):
    """Returns the environment, either the whole dungeon or windowed
    if a radius is specified. Setting @ignore_explored_visible to True
    is used when the entire dungeon needs to be revealed. However, be
    careful, as this can be CPU intensive, so @ignore_explored_visible
    set to False is the default.
    """
    env_rect = EnvironmentRect(d, radius)
    # Transpose to Diablo indexing: (width, height), instead of numpy
    # (height, weight)
    env = np.zeros((env_rect.drect.width, env_rect.drect.height),
                   dtype=np.uint16)

    for j in range(env_rect.srect.height):
        for i in range(env_rect.srect.width):
            spos = (env_rect.srect.lt[0] + i, env_rect.srect.lt[1] + j)
            obj = to_object(d, spos)
            s = 0

            if d.dFlags_np[spos] & DungeonFlag.Explored.value:
                s |= EnvironmentFlag.Explored.value
            if d.dFlags_np[spos] & DungeonFlag.Visible.value:
                s |= EnvironmentFlag.Visible.value

            if ignore_explored_visible or s & EnvironmentFlag.Explored.value:
                if is_wall(d, spos):
                    s |= EnvironmentFlag.Wall.value
                if is_trigger(d, spos):
                    s |= EnvironmentFlag.Trigger.value
                if obj is not None and is_door(obj):
                    if is_door_closed(obj):
                        s |= EnvironmentFlag.DoorClosed.value
                    else:
                        s |= EnvironmentFlag.DoorOpened.value
            if ignore_explored_visible or s & EnvironmentFlag.Visible.value:
                if d.dFlags_np[spos] & DungeonFlag.Missile.value:
                    s |= EnvironmentFlag.Missile.value
                if d.dMonster_np[spos] > 0:
                    s |= EnvironmentFlag.Monster.value

                if obj is not None:
                    if is_barrel(obj):
                        if is_breakable(obj):
                            s |= EnvironmentFlag.Barrel.value
                    elif is_crucifix(obj):
                        s |= EnvironmentFlag.Crucifix.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_chest(obj):
                        s |= EnvironmentFlag.Chest.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_sarcophagus(obj):
                        s |= EnvironmentFlag.Sarcophagus.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_door(obj):
                        # Handled above by the explored 'if' branch
                        pass
                    else:
                        s |= EnvironmentFlag.UnknownObject.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                if d.dItem_np[spos] > 0:
                    s |= EnvironmentFlag.Item.value

            if spos == player_position(d):
                s |= EnvironmentFlag.Player.value

            # Transpose to Diablo indexing: (x, y), instead of numpy (y, x)
            dpos = (env_rect.drect.lt[0] + i, env_rect.drect.lt[1] + j)
            env[dpos] = s

    return env

def get_surroundings(d, radius, t=str):
    env = get_environment(d, radius)
    surroundings = np.full(env.shape, ' ' if t == str else ord(' '), dtype=t)

    for j, row in enumerate(env):
        for i, tile in enumerate(row):
            if tile == 0:
                continue
            if tile & EnvironmentFlag.Explored.value:
                s = ' '
            if tile & EnvironmentFlag.Visible.value:
                s = '.'
            if tile & EnvironmentFlag.Wall.value:
                s = '#'
            if tile & EnvironmentFlag.Trigger.value:
                s = '$'
            if tile & EnvironmentFlag.DoorClosed.value:
                s = 'D'
            if tile & EnvironmentFlag.DoorOpened.value:
                s = 'd'
            if tile & EnvironmentFlag.Barrel.value:
                s = 'B'
            if tile & EnvironmentFlag.UnknownObject.value:
                s = 'O' if tile & EnvironmentFlag.Interactable.value else 'o'
            if tile & EnvironmentFlag.Chest.value:
                s = 'C' if tile & EnvironmentFlag.Interactable.value else 'c'
            if tile & EnvironmentFlag.Sarcophagus.value:
                s = 'S' if tile & EnvironmentFlag.Interactable.value else 's'
            if tile & EnvironmentFlag.Crucifix.value:
                s = 'U' if tile & EnvironmentFlag.Interactable.value else 'u'
            if tile & EnvironmentFlag.Item.value:
                s = 'I'
            if tile & EnvironmentFlag.Missile.value:
                s = '%'
            if tile & EnvironmentFlag.Monster.value:
                s = '@'
            if t == str and tile & EnvironmentFlag.Player.value:
                if is_player_dead(d):
                    s = 'X'
                else:
                    s = '*'
                    match player_direction(d):
                        case Direction.North.value:
                            s = "\u2191"
                        case Direction.NorthEast.value:
                            s = "\u2197"
                        case Direction.East.value:
                            s = "\u2192"
                        case Direction.SouthEast.value:
                            s = "\u2198"
                        case Direction.South.value:
                            s = "\u2193"
                        case Direction.SouthWest.value:
                            s = "\u2199"
                        case Direction.West.value:
                            s = "\u2190"
                        case Direction.NorthWest.value:
                            s = "\u2196"
            elif tile & EnvironmentFlag.Player.value:
                # XXX Simplified representation for AI
                s = 'X' if is_player_dead(d) else '*'
            surroundings[j, i] = s if t == str else ord(s)

    return surroundings

def get_dungeon_graph_and_path(d, start, goal):
    # Environment of the whole dungeon
    env = get_environment(d, ignore_explored_visible=True)
    # 0 - walls
    # 1 - empty areas, probably occupied by player, monsters, etc
    empty_env = \
        (env & EnvironmentFlag.Player.value) | \
        (env & EnvironmentFlag.Monster.value) | \
        (env & EnvironmentFlag.Barrel.value) | \
        (env & EnvironmentFlag.Item.value) | \
        (env & EnvironmentFlag.Trigger.value) | \
        (env == 0) | \
        (env == EnvironmentFlag.Explored.value) | \
        (env == EnvironmentFlag.Visible.value) | \
        (env == (EnvironmentFlag.Explored.value | \
                 EnvironmentFlag.Visible.value))
    # Doors positions
    doors = np.argwhere(env & (EnvironmentFlag.DoorOpened.value | \
                               EnvironmentFlag.DoorClosed.value))

    # Label independent regions
    labeled_regions, num_regions = maze.detect_regions(empty_env)
    # Build graph of connected regions
    regions_graph, regions_doors, doors_matrix = \
        maze.get_regions_graph(doors, labeled_regions, num_regions)

    start_region = labeled_regions[start]
    goal_region = labeled_regions[goal]

    assert start_region != 0
    assert goal_region != 0

    # Shortest path between regions
    regions_path = maze.bfs_regions_path(regions_graph, start_region,
                                         goal_region)
    assert regions_path is not None

    # Doors between regions on the shortest path. We could use set()
    # here, but we need to keep an order
    path_doors = []
    for i, region in enumerate(regions_path):
        if i < len(regions_path) - 1:
            next_region = regions_path[i + 1]
            # Get the door coordinates, which leads to the goal region
            x, y = doors_matrix[region, next_region]
            assert x != 0 and y != 0
            if (x, y) not in path_doors:
                path_doors.append((x, y))
            regions_doors[region][(x, y)] = True

    return regions_doors, labeled_regions, regions_path, path_doors

def map_shared_state(path):
    f = open(path, "r+b")
    mmapped = mmap.mmap(f.fileno(), 0)
    f.close()

    hdr = DiabloStateHeader.from_buffer(mmapped)

    class DiabloState(DiabloStateHeader):
        _fields_ = [
            ("input_queue",  ring.RingQueue),
            ("events_queue", ring.RingQueue),
            ("player",       PlayerState),
            ("game_ticks",   ctypes.c_ulonglong),
            ("game_saves",   ctypes.c_ulonglong),
            ("game_loads",   ctypes.c_ulonglong),

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
            ("ign_AvailableObjects",   ctypes.c_int32 * hdr.max_objects),
            ("ActiveObjects",          ctypes.c_int32 * hdr.max_objects),
            ("ActiveObjectCount",      ctypes.c_int32),

            #
            # Diablo
            #
            ("PauseMode",       ctypes.c_int32),

            #
            # Items
            #
            ("ign_Items",       (Item * (hdr.max_items + 1))),
            ("ign_ActiveItems", (ctypes.c_uint8 * hdr.max_items)),
            ("ActiveItemCount", ctypes.c_uint8),
            ("dItem",           (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),

            #
            # Gendung
            #
            ("ign_dTransVal",  (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dFlags",         (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("ign_dPlayer",    (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("dMonster",       (ctypes.c_int16 * hdr.maxdun[0]) * hdr.maxdun[1]),
            ("ign_dCorpse",    (ctypes.c_uint8 * hdr.maxdun[0]) * hdr.maxdun[1]),
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

    state = DiabloState.from_buffer(mmapped)

    # Create numpy arrays instead of regular
    for field_name, field_type in state._fields_:
        if field_name.startswith("ign") or field_name.startswith("padding"):
            # Ignore these fields
            continue
        field_value = getattr(state, field_name)
        if not isinstance(field_value, ctypes.Array):
            # Ignore not arrays
            continue

        np_view = np.ctypeslib.as_array(field_value)
        setattr(state, field_name + "_np", np_view)

    return mmapped, state

class DiabloGame:
    def __init__(self, state_dir, proc, log_file, mshared_path,
                 mmapped, state):
        self.state_dir = state_dir
        self.proc = proc
        self.log_file = log_file
        self.mshared_path = mshared_path
        self.mmapped = mmapped
        self.state = state
        self.last_tick = 0
        self.acceptance_tick = 0

    def __del__(self):
        self.stop_or_detach()

    def stop_or_detach(self):
        if self.proc:
            self.proc.terminate()
        if self.log_file:
            self.log_file.close()
        if self.state_dir:
            self.state_dir.cleanup()

    def ticks(self, d=None):
        t = self.state.game_ticks if d is None else d.game_ticks
        # There are two phases of tick updates:
        #     "odd" phase - before keys are processed
        #    "even" phase - after keys are processed
        # Divide on two to have a full game update cycle
        return t // 2

    def update_ticks(self):
        missed = self.ticks() - self.last_tick
        self.last_tick += missed
        return missed

    def same_ticks(self):
        diff = self.ticks() - self.last_tick
        return diff == 0

    def submit_key(self, key):
        # Busy-loop for the next tick. It's important to understand that every
        # key has press and release phases, so we can't submit the same
        # two keys sequentially if one was not released; otherwise, the
        # second key will be lost. With the following loop, we
        # introduce a 1-tick delay from the previous key acceptance.
        # Also be aware that each tick has two phases, hence the +2.
        while self.state.game_ticks < self.acceptance_tick + 2:
            time.sleep(0.01)

        entry = self.state.input_queue.get_entry_to_submit()
        assert entry
        entry.type = key
        entry.data = 0

        # Submit key
        self.state.input_queue.submit()

        # Busy-loop for actual key acceptance
        while self.state.input_queue.nr_submitted_entries() != 0:
            time.sleep(0.01)

        # Acceptance of a key is always the last "even" phase, thus
        # round up on 2.
        self.acceptance_tick = round_up_int(self.state.game_ticks, 2)

    @staticmethod
    def attach_mshared_path(mshared_path):
        for attempt in range(0, 10):
            try:
                # Open the file and map it to memory
                mmapped, state = map_shared_state(mshared_path)
                return mmapped, state
            except FileNotFoundError:
                time.sleep(0.1)
        else:
            raise FileNotFoundError(mshared_path)

    @staticmethod
    def run(config):
        cfg_file = open("diablo.ini.template", "r")
        cfg = cfg_file.read()
        cfg_file.close()
        cfg = cfg.format(seed=config["seed"],
                         headless=config["headless"],
                         mshared_filename=config["mshared_filename"],
                         no_monsters=config["no_monsters"])

        prefix = "diablo-%d-" % config["seed"]
        state_dir = tempfile.TemporaryDirectory(prefix=prefix)
        cfg_file = open(state_dir.name + "/diablo.ini", "w")
        cfg_file.write(cfg)
        cfg_file.close()

        log_file = open(state_dir.name + "/diablo.log", "w", buffering=1)

        cmd = [
            config["diablo-bin-path"],
            '-n', # Skip startup videos (if not headless)
            '-f', # Display frames per second (if not headless)
            '--config-dir', state_dir.name,
            '--save-dir', state_dir.name,
            '--data-dir', config["diablo-data-path"]
        ]
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        mshared_path = os.path.abspath(state_dir.name + "/" + config["mshared_filename"])
        mmapped, state = DiabloGame.attach_mshared_path(mshared_path)
        return DiabloGame(state_dir, proc, log_file, mshared_path, mmapped, state)

    @staticmethod
    def attach(mshared_path):
        mmapped, state = DiabloGame.attach_mshared_path(mshared_path)
        return DiabloGame(None, None, None, mshared_path, mmapped, state)

    @staticmethod
    def run_or_attach(config):
        if 'attach-path' in config:
            return DiabloGame.attach(config['attach-path'])
        return DiabloGame.run(config)
