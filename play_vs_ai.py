"""Play Golf vs 4 AI opponents in the terminal.

Human always plays seat 0. Run:
    python play_vs_ai.py [path_to_model.pt]
"""

import sys
import torch

from src.consts import (
    OBS_SIZE, NUM_PLAYERS, FACE_STR_MAP, POINT_VALUES,
    STAGE_ARRANGE, STAGE_FLIP1, STAGE_FLIP2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD,
    NUM_OPTIONS_PER_R, PARTITIONS_PER_R, COLUMNS, TOTAL_CARDS,
)
from src.model import ValueNet
from src.vector_env import VectorGolfEnv
from src.decision import make_decisions


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUMAN_SEAT = 0
EID = 0  # Single-env play

STAGE_NAMES = {
    STAGE_ARRANGE: "ARRANGE",
    STAGE_FLIP1: "FLIP 1",
    STAGE_FLIP2: "FLIP 2",
    STAGE_PLAY_DRAW: "PLAY (first choice)",
    STAGE_PLAY_DISCARD: "PLAY (second choice)",
}


# ---------- card formatting ----------
def card_str(card_id):
    """0-103 → 'AR', '5B', etc."""
    color = card_id // 52
    face = card_id % 13
    return FACE_STR_MAP[face] + ("R" if color == 0 else "B")


def card_color_only(card_id):
    """For face-down cards we know the color. Returns 'R?' or 'B?'."""
    return ("R?" if card_id // 52 == 0 else "B?")


def cell_str(card_id, visible):
    if visible:
        return card_str(card_id)
    return card_color_only(card_id)


def card_pts(card_id):
    return POINT_VALUES[card_id % 13]


def player_label(p):
    return "YOU" if p == HUMAN_SEAT else f"AI {p}"


# ---------- rendering ----------
def render_hand(hand, visible, label, placed=True, is_acting=False, is_finisher=False, took_final=False):
    """Render a 3x3 grid for one player. Pre-arrangement: cards shown in deal
    order (just colors). Post-arrangement: real grid with face-up reveals."""
    h = hand.cpu().tolist()
    v = visible.cpu().tolist()

    flags = []
    if is_acting:
        flags.append("ACTING")
    if is_finisher:
        flags.append("FINISHER")
    if took_final:
        flags.append("FINAL TURN TAKEN")
    flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""

    if not placed:
        # Show in deal order, colors only — no grid layout yet.
        cells = [card_color_only(h[i]) for i in range(9)]
        return (
            f"  {label}{flag_str}  (not arranged yet)\n"
            f"     deal: {' '.join(cells)}"
        )

    cells = [cell_str(h[s], v[s]) for s in range(9)]
    return (
        f"  {label}{flag_str}\n"
        f"     [0] {cells[0]}    [1] {cells[1]}    [2] {cells[2]}\n"
        f"     [3] {cells[3]}    [4] {cells[4]}    [5] {cells[5]}\n"
        f"     [6] {cells[6]}    [7] {cells[7]}    [8] {cells[8]}"
    )


def render_state(env, history, header=""):
    print()
    print("=" * 72)
    if header:
        print(f"  {header}")
        print()
    if history:
        print("  Recent actions:")
        for line in history[-6:]:
            print(f"    • {line}")
        print()

    cur = env.current_player_idx[EID].item()
    cur_stage = env.stages[EID, cur].item()
    finisher = env.finisher_idx[EID].item()
    final_turns = env.final_turns_taken[EID].cpu().tolist()
    top_disc = env.top_discard[EID].item()
    cards_left = max(0, TOTAL_CARDS - env.draw_ptr[EID].item())

    if cur_stage == STAGE_PLAY_DISCARD:
        print(f"  REVEALED CARD from draw pile (now top of discard): "
              f"{card_str(top_disc)} ({card_pts(top_disc):+d} pts)")
        print(f"  DRAW PILE: {cards_left} cards remaining")
    else:
        # In setup or PLAY_DRAW: show top discard + top of draw color
        next_ptr = env.draw_ptr[EID].item()
        next_color_str = "—"
        if next_ptr < TOTAL_CARDS:
            next_card = env.deck_cards[EID, next_ptr].item()
            next_color_str = "R" if next_card < 52 else "B"
        print(f"  TOP DISCARD: {card_str(top_disc)} ({card_pts(top_disc):+d} pts)"
              f"     DRAW PILE: {cards_left} cards (next color: {next_color_str})")

    if finisher != -1:
        ft_strs = " ".join(
            f"{player_label(i)}={'✓' if final_turns[i] else '·'}"
            for i in range(NUM_PLAYERS)
        )
        print(f"  FINISHER: {player_label(finisher)}     Final turns: {ft_strs}")

    print()
    for p in range(NUM_PLAYERS):
        p_stage = env.stages[EID, p].item()
        placed = (p_stage > STAGE_ARRANGE)
        print(render_hand(
            env.hands[EID, p],
            env.visible[EID, p],
            player_label(p),
            placed=placed,
            is_acting=(p == cur),
            is_finisher=(p == finisher),
            took_final=final_turns[p],
        ))
        print()


# ---------- human input handlers ----------
def _input(prompt):
    while True:
        try:
            return input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            sys.exit(0)


def get_human_arrange_action(hand):
    """Show the available partition options for human's R; return option index (0..NUM_OPTIONS-1)."""
    h = hand.cpu().tolist()
    is_red = [c < 52 for c in h]
    R = sum(is_red)
    options = PARTITIONS_PER_R[R]
    print(f"  Your colors (deal order): {' '.join('R?' if ir else 'B?' for ir in is_red)}")
    print(f"  Reds: {R}, Blues: {9 - R}")
    print(f"  Available arrangements (red counts per column):")
    for i, p in enumerate(options):
        print(f"    [{i}] columns: {p[0]}R / {p[1]}R / {p[2]}R    "
              f"(blues fill the rest)")
    while True:
        s = _input(f"  Pick arrangement [0-{len(options)-1}]: ")
        try:
            n = int(s)
            if 0 <= n < len(options):
                return n
        except ValueError:
            pass
        print("    Invalid.")


def get_human_flip_slot(visible):
    """Pick a face-down slot (0-8). Already-flipped slots are rejected."""
    v = visible.cpu().tolist()
    fd = [i for i, vis in enumerate(v) if not vis]
    print(f"  Face-down slots available: {fd}")
    while True:
        s = _input("  Pick a slot to flip (0-8): ")
        try:
            n = int(s)
            if n in fd:
                return n
        except ValueError:
            pass
        print("    Invalid (must be a face-down slot).")


def get_human_play_action(stage):
    if stage == STAGE_PLAY_DRAW:
        prompt = ("  YOUR TURN. Enter slot 0-8 to take TOP DISCARD, "
                  "or P to pass and peek at the draw pile: ")
    else:
        prompt = ("  Enter slot 0-8 to take the REVEALED CARD, "
                  "or P to pass (it stays on discard): ")
    while True:
        s = _input(prompt)
        if s in ("p", "pass", "9"):
            return 9
        try:
            n = int(s)
            if 0 <= n <= 8:
                return n
        except ValueError:
            pass
        print("    Invalid. Enter a digit 0-8 or P.")


# ---------- action descriptions ----------
def describe_action(action, stage, prev_top, prev_hand, prev_visible, player, env):
    """One-line summary of what just happened. Called AFTER env.step()."""
    label = player_label(player)
    if stage == STAGE_ARRANGE:
        return f"{label} arranged their hand (option {action})."
    if stage == STAGE_FLIP1:
        # The slot just flipped; show what was revealed.
        revealed = card_str(env.hands[EID, player, action].item())
        return f"{label} flipped slot {action} → {revealed} (first flip)."
    if stage == STAGE_FLIP2:
        revealed = card_str(env.hands[EID, player, action].item())
        return f"{label} flipped slot {action} → {revealed} (second flip)."
    # PLAY stages
    if action == 9:
        if stage == STAGE_PLAY_DRAW:
            return f"{label} passed on {card_str(prev_top)} → drew from deck."
        else:
            return f"{label} passed on drawn card {card_str(prev_top)} (it stays on discard)."
    taken = card_str(prev_top)
    source = "discard" if stage == STAGE_PLAY_DRAW else "draw"
    replaced = card_str(prev_hand[action].item())
    return (f"{label} took {taken} from {source} → slot {action} (replaced {replaced}).")


# ---------- main ----------
def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "latest_model.pt"

    print(f"Loading {model_path}...")
    network = ValueNet(OBS_SIZE, num_players=NUM_PLAYERS).to(DEVICE)
    network.load_state_dict(torch.load(model_path, map_location=DEVICE))
    network.eval()

    env = VectorGolfEnv(num_envs=1, device=DEVICE)
    history = []

    dealer = env.dealer_idx[EID].item()
    first = env.current_player_idx[EID].item()
    print()
    print("=" * 72)
    print("  GOLF — you are seat 0 (YOU). AI plays seats 1, 2, 3, 4.")
    print(f"  Dealer: {player_label(dealer)}     First to act: {player_label(first)}")
    print("=" * 72)

    final_info = None
    while True:
        cur = env.current_player_idx[EID].item()
        stage = env.stages[EID, cur].item()
        prev_top = env.top_discard[EID].item()
        prev_hand = env.hands[EID, cur].clone()
        prev_visible = env.visible[EID, cur].clone()

        if cur == HUMAN_SEAT:
            # Render only at the start of OUR turn-segments to avoid clutter.
            # Setup phase: render at ARRANGE start, then sub-steps share that view.
            if stage in (STAGE_ARRANGE, STAGE_PLAY_DRAW) or stage == STAGE_FLIP1:
                render_state(env, history,
                             header=f"YOUR TURN — {STAGE_NAMES[stage]}")

            if stage == STAGE_ARRANGE:
                action = get_human_arrange_action(prev_hand)
            elif stage in (STAGE_FLIP1, STAGE_FLIP2):
                action = get_human_flip_slot(prev_visible)
            else:
                action = get_human_play_action(stage)
        else:
            # Render once per AI turn at its first sub-step (ARRANGE or PLAY_DRAW).
            if stage in (STAGE_ARRANGE, STAGE_PLAY_DRAW):
                render_state(env, history,
                             header=f"{player_label(cur)} thinking ({STAGE_NAMES[stage]})...")
            actions = make_decisions(env, network, epsilon=0.0)
            action = actions[EID].item()

        actions_tensor = torch.tensor([action], device=DEVICE)
        _next_obs, _dones, _acting, info = env.step(actions_tensor)
        history.append(describe_action(action, stage, prev_top, prev_hand, prev_visible, cur, env))

        if "all_scores" in info:
            final_info = info
            break

    # --- Game over ---
    print()
    print("=" * 72)
    print("  GAME OVER")
    print("=" * 72)
    print()
    print("  Action log (last 12):")
    for line in history[-12:]:
        print(f"    • {line}")
    print()

    final_hands = final_info["final_hands"][0]
    final_visible = final_info["final_visible"][0]
    final_scores = final_info["all_scores"][0]
    min_score = final_scores.min().item()

    for p in range(NUM_PLAYERS):
        sc = final_scores[p].item()
        winner_tag = "  ← WINNER" if sc == min_score else ""
        print(render_hand(final_hands[p], final_visible[p], player_label(p), placed=True))
        print(f"     SCORE: {sc}{winner_tag}")
        print()

    if final_scores[HUMAN_SEAT].item() == min_score:
        n_winners = (final_scores == min_score).sum().item()
        if n_winners == 1:
            print("  You won.")
        else:
            print(f"  Tied for best ({n_winners} winners).")
    else:
        gap = final_scores[HUMAN_SEAT].item() - min_score
        print(f"  You scored {final_scores[HUMAN_SEAT].item()}; best was {min_score} "
              f"(gap of {gap}).")
    print()


if __name__ == "__main__":
    main()
