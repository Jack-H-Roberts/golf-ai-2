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

# Cards 0-51 are red deck, 52-103 are blue deck. card // 52 = color, card % 13 = face.
RED = 0
BLUE = 1

# --- STAGE CONSTANTS ---
# 5 stages, encoded as one-hot in obs (whose stage = the acting player's stage).
STAGE_ARRANGE = 0       # Choose how to lay out R reds and 9-R blues across the 3x3.
STAGE_FLIP1 = 1         # Choose first slot to flip.
STAGE_FLIP2 = 2         # Choose second slot to flip (must differ from FLIP1's slot).
STAGE_PLAY_DRAW = 3     # First play choice: see top discard, decide take/pass.
STAGE_PLAY_DISCARD = 4  # Second play choice: passed already, see drawn card, decide take/pass.
NUM_STAGES = 5

# --- COLUMN INDICES (3x3 grid arranged 0..8 row-major) ---
# Columns are vertical: indices [0,3,6], [1,4,7], [2,5,8]
COLUMNS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]

# --- ARRANGEMENT PARTITION TABLE ---
# Maps (R, action_index) → partition (k0, k1, k2) reds across columns.
# Partitions are sorted-decreasing canonical form; columns are interchangeable
# so picking any specific (k0, k1, k2) covers all column-permuted equivalents.
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
NUM_OPTIONS_PER_R = [len(PARTITIONS_PER_R[r]) for r in range(10)]  # [1,1,2,3,3,3,3,2,1,1]


def _build_red_slot_mask_table():
    """Returns a [10, 3, 9] long tensor: red_slot_mask[R, k, slot] = 1 if that
    slot holds a red card under PARTITIONS_PER_R[R][k]. For k >= NUM_OPTIONS_PER_R[R]
    we clamp to the last valid partition so out-of-range actions are well-defined.
    """
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


# --- OBSERVATION SIZES (891 dims, deal-order global, no ego rotation) ---
#   5    whose-turn one-hot (deal order)
#   5    finisher one-hot (deal order; all-zero if no finisher yet)
#   5    has-taken-final-turn bitmap (deal order)
#   5    acting player's stage one-hot (5 stages)
#   1    remaining reds across players-after-acting still in ARRANGE (sum/9)
#   26   graveyard distribution: 13 P(face|red) + 13 P(face|blue) over unseen pool
#   17   top discard:  2 color one-hot + 13 face one-hot + 1 norm point value + 1 visibility (=1)
#   17   top draw:     2 color one-hot + 13 face dist (from grav[color]) + 1 value (=0) + 1 visibility (=0)
#   765  hand cards (5 players * 9 cards * 17 dims, deal order)
#         per card (placed):    2 color + 13 face (one-hot if visible, P(face|color) if face-down) + 1 value + 1 visibility
#         per card (unplaced):  all 17 dims = 0
#   15   per-column expected score (5 players * 3 columns); 0 for unplaced players
#   15   per-column 3-of-a-kind probability (5 players * 3 columns); 0 for unplaced players
#   5    per-player total hand expected score (deal order); 0 for unplaced players
#   5    per-player face-down count (deal order, normalized /9); 0 for unplaced players
#   5    per-player score gap from leader (own_EV - min_EV); 0 for unplaced players
OBS_SIZE = 891
ACTION_SIZE = 10           # 9 slot replacements / partition options + 1 pass
VALUE_OUTPUT_DIM = NUM_PLAYERS  # 5-dim P(player wins) softmax

# 4 cards of each (color, face) in 2 standard decks (no jokers).
INITIAL_PER_COLOR_FACE = 4
