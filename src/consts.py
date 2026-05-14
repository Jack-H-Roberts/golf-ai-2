import numpy as np

# --- GAME CONFIG ---
NUM_PLAYERS = 5
CARDS_PER_PLAYER = 9
NUM_DECKS = 2
TOTAL_CARDS = 52 * NUM_DECKS

# --- CARD MAPPING ---
FACE_ACE = 0
FACE_2 = 1
FACE_3 = 2
FACE_4 = 3
FACE_5 = 4
FACE_6 = 5
FACE_7 = 6
FACE_8 = 7
FACE_9 = 8
FACE_10 = 9
FACE_JACK = 10
FACE_QUEEN = 11
FACE_KING = 12

POINT_VALUES = {
    FACE_ACE: 1, FACE_2: 2, FACE_3: 3, FACE_4: 4,
    FACE_5: 5, FACE_6: 6, FACE_7: 7, FACE_8: -2,
    FACE_9: 9, FACE_10: 10, FACE_JACK: 10,
    FACE_QUEEN: 10, FACE_KING: 0
}

FACE_STR_MAP = {
    FACE_ACE: "A", FACE_2: "2", FACE_3: "3", FACE_4: "4",
    FACE_5: "5", FACE_6: "6", FACE_7: "7", FACE_8: "8",
    FACE_9: "9", FACE_10: "T", FACE_JACK: "J",
    FACE_QUEEN: "Q", FACE_KING: "K"
}

# Cards 0-51 are red, 52-103 are blue. card // 52 = color, card % 13 = face.
RED = 0
BLUE = 1

# --- STAGE CONSTANTS ---
STAGE_ARRANGE = 0
STAGE_FLIP1 = 1
STAGE_FLIP2 = 2
STAGE_PLAY_DRAW = 3
STAGE_PLAY_DISCARD = 4
NUM_STAGES = 5

# --- COLUMN INDICES (3x3 grid, 0..8 row-major) ---
# Column 0 = slots 0, 3, 6 (vertical left); Column 1 = 1, 4, 7; Column 2 = 2, 5, 8.
# Bijection: slot = pos*3 + col, where pos = slot // 3 (row), col = slot % 3.
COLUMNS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]

# --- ARRANGEMENT PARTITION TABLE ---
PARTITIONS_PER_R = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0)],
    2: [(2, 0, 0), (1, 1, 0)],
    3: [(3, 0, 0), (2, 1, 0), (1, 1, 1)],
    4: [(3, 1, 0), (2, 2, 0), (2, 1, 1)],
    5: [(3, 2, 0), (3, 1, 1), (2, 2, 1)],
    6: [(3, 3, 0), (3, 2, 1), (2, 2, 2)],
    7: [(3, 3, 1), (3, 2, 2)],
    8: [(3, 3, 2)],
    9: [(3, 3, 3)],
}
MAX_ARRANGEMENT_OPTIONS = 3
NUM_OPTIONS_PER_R = [len(PARTITIONS_PER_R[r]) for r in range(10)]


def _build_red_slot_mask_table():
    import torch
    table = torch.zeros((10, MAX_ARRANGEMENT_OPTIONS, 9), dtype=torch.long)
    for r in range(10):
        partitions = PARTITIONS_PER_R[r]
        for k, (k0, k1, k2) in enumerate(partitions):
            mask = [0] * 9
            for col_idx, ki in enumerate([k0, k1, k2]):
                for j in range(ki):
                    mask[COLUMNS[col_idx][j]] = 1
            table[r, k] = torch.tensor(mask, dtype=torch.long)
        last = len(partitions) - 1
        for k in range(len(partitions), MAX_ARRANGEMENT_OPTIONS):
            table[r, k] = table[r, last]
    return table


# --- CANONICALIZATION ALPHABET ---
# 15-letter ordering, ascending: K=0, Q=1, J=2, 10=3, 9=4, 8=5, 7=6, 6=7, 5=8,
# 4=9, 3=10, 2=11, A=12, B-down=13, R-down=14.
# Visible: alphabet = 12 - face (so K(face=12)→0, A(face=0)→12).
# Face-down: 13 if blue, 14 if red.
# Smallest at canonical slot 0 (top-left); largest at slot 8 (bottom-right).
ALPHABET_BDOWN = 13
ALPHABET_RDOWN = 14
ALPHABET_SIZE = 15
COL_KEY_BASE = ALPHABET_SIZE         # 15
COL_KEY_BASE_SQ = COL_KEY_BASE ** 2  # 225

# --- OBSERVATION LAYOUT (OBS_SIZE = 886) ---
# Post-ego-rotation: per-player blocks ordered with acting=0, then dealer-cycle.
# whose-turn one-hot dropped (always [1,0,0,0,0] after rotation).
# Per-card encoding (17 dims): color bits zeroed for visible cards (visible-color
# collapse — color is redundant given the graveyard block).
#
#   5    finisher one-hot                  [0:5]      ← rotated
#   5    has-taken-final-turn              [5:10]     ← rotated
#   5    acting player's stage             [10:15]
#   1    remaining reds                    [15:16]
#   26   graveyard distribution            [16:42]    (13 red, 13 blue) ← color block
#   17   top discard                       [42:59]    (2 color + 13 face + 1 val + 1 vis)
#   17   top draw                          [59:76]
#   765  hand cards (5 * 9 * 17)           [76:841]   ← rotated; visible-color collapsed
#   15   per-column expected score         [841:856]  ← rotated
#   15   per-column 3-of-a-kind prob       [856:871]  ← rotated
#   5    per-player hand EV                [871:876]  ← rotated
#   5    per-player face-down count        [876:881]  ← rotated
#   5    per-player score gap              [881:886]  ← rotated
OBS_OFFSET_FINISHER = 0
OBS_OFFSET_FINAL_TURNS_TAKEN = 5
OBS_OFFSET_STAGE = 10
OBS_OFFSET_REMAINING_REDS = 15
OBS_OFFSET_GRAVEYARD = 16
OBS_OFFSET_TOP_DISCARD = 42
OBS_OFFSET_TOP_DRAW = 59
OBS_OFFSET_HAND_CARDS = 76
OBS_OFFSET_COL_EV = 841
OBS_OFFSET_COL_MATCH = 856
OBS_OFFSET_HAND_EV = 871
OBS_OFFSET_FD_COUNT = 876
OBS_OFFSET_SCORE_GAP = 881

OBS_HAND_CARD_DIM = 17
OBS_NUM_HAND_CARDS = NUM_PLAYERS * 9

OBS_SIZE = 886
ACTION_SIZE = 10
VALUE_OUTPUT_DIM = NUM_PLAYERS

INITIAL_PER_COLOR_FACE = 4
