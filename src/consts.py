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

# --- OBSERVATION SIZES ---
# 728-dim obs (simplified env, new encoding):
#   1   stage flag (PLAY_DRAW=1 vs PLAY_DISCARD=0)
#   5   relative finisher one-hot
#   26  graveyard distribution: 13 P(face|red) + 13 P(face|blue), normalized over unseen pool
#   15  top discard:  1 color + 13 face one-hot + 1 normalized point value (always visible)
#   1   top draw color (face is unknown; distribution comes from graveyard[color])
#   675 hand cards (45 cards x 15 dims):
#         1 color + 13 face (one-hot if visible, all-zero if face-down) + 1 value (or 0)
#   5   per-player face-down count, ego-rotated, normalized /9
OBS_SIZE = 728
ACTION_SIZE = 10

# 4 cards of each (color, face) in 2 standard decks (no jokers).
INITIAL_PER_COLOR_FACE = 4

# --- STAGE CONSTANTS ---
STAGE_ARRANGE = 0
STAGE_FLIP_1 = 1
STAGE_FLIP_2 = 2
STAGE_PLAY_DRAW = 3
STAGE_PLAY_DISCARD = 4
