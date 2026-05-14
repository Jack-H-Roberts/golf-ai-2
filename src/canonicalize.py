"""Canonicalization and ego rotation for Golf state.

Two pure functions that vector_env.py imports and applies to hand/visibility
tensors. Both are vectorized over arbitrary leading batch dims.

CANONICALIZATION
================
Sort each player's 9-card hand into a unique canonical form, breaking the
symmetry between equivalent hands. The canonical form depends only on the set
of cards and their visibility — not on the order they were dealt.

Alphabet ordering (ascending):
    K=0, Q=1, J=2, 10=3, 9=4, 8=5, 7=6, 6=7, 5=8, 4=9, 3=10, 2=11, A=12,
    B-down=13, R-down=14
Equivalently: alphabet = 12 - face for visible cards; for face-down cards,
13 if blue, 14 if red.

Two-level sort:
  (1) Sort each column ascending by per-card alphabet key.
  (2) Sort the three columns ascending by a base-15 polynomial of their
      (now-sorted) 3 card keys: col_key = k0*225 + k1*15 + k2.

Smallest key ends up at canonical slot 0 (top-left); largest at slot 8
(bottom-right). So canonical hands have Ks/visible-low-cards toward the
top-left and face-down reds toward the bottom-right.

Stable sort breaks ties by input order — important for the env where the
underlying card identities matter (the env tracks card_ids; the network only
sees the canonical form).


EGO ROTATION
============
Rotate a [N, NUM_PLAYERS, ...] tensor so the acting player sits at index 0,
followed by dealer-cycle order. After rotation, position i holds the data
for player (acting + i) % NUM_PLAYERS.

The whose-turn one-hot becomes redundant (always [1,0,0,0,0]) and is dropped
from the obs. The finisher one-hot still varies per game state; it's encoded
at position (finisher - acting) % NUM_PLAYERS, with zero-out if no finisher.
"""

import torch
from .consts import NUM_PLAYERS, ALPHABET_BDOWN, COL_KEY_BASE, COL_KEY_BASE_SQ


# Slot-to-column mapping for a 3x3 grid with row-major slot indices.
# Column 0 = slots 0, 3, 6; column 1 = 1, 4, 7; column 2 = 2, 5, 8.
# Bijection: slot = pos*3 + col, where pos = slot // 3 and col = slot % 3.
_COL_IDX_CACHE = {}


def _col_idx(device):
    key = (device.type, getattr(device, 'index', None))
    cached = _COL_IDX_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([[0, 3, 6], [1, 4, 7], [2, 5, 8]],
                              device=device, dtype=torch.long)
        _COL_IDX_CACHE[key] = cached
    return cached


def canonicalize_hands(hands, visible):
    """Sort each hand into canonical form.

    Args:
        hands: [..., 9] long tensor of card IDs in [0, 103].
               Card 0-51 = red, 52-103 = blue. face = card % 13.
        visible: [..., 9] bool tensor.

    Returns:
        (hands_canonical, visible_canonical) — same shape, sorted.
    """
    col_idx = _col_idx(hands.device)

    # Per-card alphabet key
    face = hands % 13
    is_blue_long = (hands >= 52).long()
    # Visible: alphabet = 12 - face  (K=0 ... A=12)
    # Face-down: 13 if blue, 14 if red
    keys = torch.where(
        visible,
        12 - face,
        ALPHABET_BDOWN + (1 - is_blue_long),
    )

    # Gather into [..., 3 cols, 3 pos]
    hands_2d = hands[..., col_idx]
    visible_2d = visible[..., col_idx]
    keys_2d = keys[..., col_idx]

    # Sort within columns (last dim)
    sort_within = keys_2d.argsort(dim=-1, stable=True)
    hands_2d = hands_2d.gather(-1, sort_within)
    visible_2d = visible_2d.gather(-1, sort_within)
    keys_2d = keys_2d.gather(-1, sort_within)

    # Column key from 3 sorted card keys (base-15 polynomial)
    col_keys = (keys_2d[..., 0] * COL_KEY_BASE_SQ
                + keys_2d[..., 1] * COL_KEY_BASE
                + keys_2d[..., 2])

    # Sort columns ascending
    sort_cols = col_keys.argsort(dim=-1, stable=True)
    sort_cols_3d = sort_cols.unsqueeze(-1).expand(*sort_cols.shape, 3)
    hands_2d = hands_2d.gather(-2, sort_cols_3d)
    visible_2d = visible_2d.gather(-2, sort_cols_3d)

    # Reshape back to flat [..., 9] in slot order. Recall slot = pos*3 + col,
    # and hands_2d has axes [col, pos]; transposing gives [pos, col] which
    # flattens row-major to the slot order we need.
    hands_canonical = hands_2d.transpose(-1, -2).reshape(hands.shape)
    visible_canonical = visible_2d.transpose(-1, -2).reshape(visible.shape)
    return hands_canonical, visible_canonical


def ego_rotate(tensor, acting):
    """Rotate a [N, NUM_PLAYERS, ...] tensor so acting sits at index 0.

    Args:
        tensor: leading shape [N, NUM_PLAYERS, ...].
        acting: [N] long tensor of acting-player indices in [0, NUM_PLAYERS).

    Returns:
        Rotated tensor, same shape.

    After rotation, T_rotated[n, i] = tensor[n, (acting[n] + i) % NUM_PLAYERS].
    Applies to any trailing shape and dtype via advanced indexing.
    """
    N = tensor.shape[0]
    device = tensor.device
    i_arange = torch.arange(NUM_PLAYERS, device=device)
    rotated_indices = (acting.unsqueeze(1) + i_arange.unsqueeze(0)) % NUM_PLAYERS
    e_idx = torch.arange(N, device=device).view(N, 1).expand(N, NUM_PLAYERS)
    return tensor[e_idx, rotated_indices]


def rotated_player_index(target_player_idx, acting):
    """Where does `target_player_idx` end up after ego rotation?

    Returns (target - acting) % NUM_PLAYERS. Use this for things like the
    finisher one-hot: in the rotated obs, the finisher's slot is
    rotated_player_index(finisher_idx, current_player_idx).
    """
    return (target_player_idx - acting) % NUM_PLAYERS
