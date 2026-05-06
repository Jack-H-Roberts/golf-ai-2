import torch
import numpy as np
import time
from colorama import Fore, Style, init

from src.golf_game import GolfGame, STAGE_ARRANGE, STAGE_FLIP_1, STAGE_FLIP_2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD
from src.model import GolfNet
from src.consts import *

init(autoreset=True)

MODEL_PATH = "latest_model.pt"


# --- ADAPTER: BUILD 736-DIM OBSERVATION (matches VectorGolfEnv.get_obs) ---
def build_single_observation(game, p_idx, device):
    """
    Constructs the 736-float input vector for a single player from the GolfGame state,
    matching the layout produced by VectorGolfEnv.get_obs.
    """
    obs_list = []

    # 1. Stage (5)
    stage = game.player_stages[p_idx]
    st_oh = np.zeros(5)
    st_oh[stage] = 1.0
    obs_list.extend(st_oh)

    # 2. Position (5)
    pos_oh = np.zeros(NUM_PLAYERS)
    pos_oh[p_idx] = 1.0
    obs_list.extend(pos_oh)

    # Rotation indices: relative to "us"
    rot_idxs = [(p_idx + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]

    # 3. Red counts per player (5)
    counts = []
    for i in rot_idxs:
        r_count = int(np.sum(game.hands[i] < 52))
        counts.append(r_count)
    obs_list.extend(counts)

    # 4. Finisher one-hot (5)
    fin_oh = np.zeros(5)
    if game.finisher_idx != -1:
        rel_fin = (game.finisher_idx - p_idx + NUM_PLAYERS) % NUM_PLAYERS
        fin_oh[rel_fin] = 1.0
    obs_list.extend(fin_oh)

    # 5. Top discard: color (1) + face (13)
    if len(game.discard_pile) > 0:
        td = game.discard_pile[-1]
        is_blue = 1.0 if td >= 52 else 0.0
        face_oh = np.zeros(13)
        face_oh[td % 13] = 1.0
    else:
        is_blue = 0.0
        face_oh = np.zeros(13)
    obs_list.append(is_blue)
    obs_list.extend(face_oh)

    # 6. Draw pile color (1)
    if len(game.draw_pile) > 0:
        draw_top = game.draw_pile[-1]
        is_d_blue = 1.0 if draw_top >= 52 else 0.0
    else:
        is_d_blue = 0.0
    obs_list.append(is_d_blue)

    # 7. Graveyard counts (26)
    graveyard = np.zeros(26)
    dead_cards = game.discard_pile[:-1]
    for c in dead_cards:
        is_b = (c >= 52)
        face = c % 13
        idx = (13 if is_b else 0) + face
        graveyard[idx] += 1.0
    obs_list.extend(graveyard)

    # 8. Known hands (5 players * 9 cards * 15 bits = 675)
    for i in rot_idxs:
        hand = game.hands[i]
        vis = game.visible[i]
        stage_i = game.player_stages[i]
        for j in range(9):
            card_vec = np.zeros(15)
            c_val = hand[j]
            is_b = c_val >= 52
            face = c_val % 13
            # Color shown if visible OR past arrange phase
            show_color = bool(vis[j]) or (stage_i > STAGE_ARRANGE)
            if show_color:
                card_vec[1 if is_b else 0] = 1.0
            # Face shown only if visible
            if bool(vis[j]):
                card_vec[2 + face] = 1.0
            obs_list.extend(card_vec)

    arr = np.array(obs_list, dtype=np.float32)
    assert arr.shape[0] == OBS_SIZE, f"Built obs of size {arr.shape[0]}, expected {OBS_SIZE}"
    return torch.from_numpy(arr).unsqueeze(0).to(device)


# --- HUMAN INTERACTION ---
def human_arrange_phase(game):
    p_idx = 0
    hand = game.hands[p_idx]

    red_cards = [c for c in hand if c < 52]
    blue_cards = [c for c in hand if c >= 52]
    num_reds = len(red_cards)

    game.render()
    print(f"{Fore.CYAN}--- ARRANGEMENT PHASE ---{Style.RESET_ALL}")
    print(f"You have {num_reds} Red cards and {len(blue_cards)} Blue cards.")

    if num_reds == 0:
        print("No Red cards to place. Proceeding automatically.")
        game.player_stages[p_idx] = STAGE_FLIP_1
        return

    while True:
        print(f"Enter {num_reds} indices (0-8) to place your Red cards.")
        print("Example: '0 4 8' puts Reds in top-left, center, bottom-right.")
        choice = input(">> ").strip()

        try:
            indices = list(map(int, choice.split()))
            if len(indices) != num_reds:
                print(f"{Fore.RED}Need {num_reds} indices, got {len(indices)}.{Style.RESET_ALL}")
                continue
            if any(i < 0 or i > 8 for i in indices):
                print(f"{Fore.RED}Indices must be 0-8.{Style.RESET_ALL}")
                continue
            if len(set(indices)) != len(indices):
                print(f"{Fore.RED}Duplicate indices.{Style.RESET_ALL}")
                continue

            new_hand = np.zeros(9, dtype=int)
            for i, r_val in zip(indices, red_cards):
                new_hand[i] = r_val
            remaining_slots = [s for s in range(9) if s not in indices]
            for i, b_val in zip(remaining_slots, blue_cards):
                new_hand[i] = b_val

            game.hands[p_idx] = new_hand
            print(f"{Fore.GREEN}Arrangement accepted.{Style.RESET_ALL}")
            game.player_stages[p_idx] = STAGE_FLIP_1
            break

        except ValueError:
            print(f"{Fore.RED}Numbers separated by spaces only.{Style.RESET_ALL}")


def get_human_move(game):
    stage = game.player_stages[0]
    valid_moves = []
    if stage in [STAGE_FLIP_1, STAGE_FLIP_2]:
        valid_moves = [i for i in range(9) if not game.visible[0][i]]
    elif stage in [STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD]:
        valid_moves = list(range(10))

    print(f"{Fore.GREEN}Your turn. Valid: {valid_moves}{Style.RESET_ALL}")
    while True:
        try:
            val = int(input("Enter action (0-8 or 9): "))
            if val in valid_moves:
                return val
            print("Invalid move.")
        except ValueError:
            print("Numbers only.")


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading AI...")

    agent = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        agent.eval()
        print("Model loaded.")
    except FileNotFoundError:
        print("No model found. Run training first.")
        return

    game = GolfGame()
    print("Welcome to Golf AI. You are Player 0.")
    time.sleep(1)

    while True:
        curr_p = game.current_turn_idx
        stage = game.player_stages[curr_p]

        if curr_p == 0:
            if stage == STAGE_ARRANGE:
                human_arrange_phase(game)
                continue
            game.render()
            action = get_human_move(game)
            _, _, _, msg = game.step(action)
            print(f"You: {msg}")
            time.sleep(1)
        else:
            if stage == STAGE_ARRANGE:
                # Note: golf_game.py arrange phase is a coarse random shuffle.
                # The trained network expects an incremental arrangement that
                # this single-player game class doesn't support; the AI here
                # is approximate during arrange.
                _, _, _, msg = game.step(0)
            else:
                obs_tensor = build_single_observation(game, curr_p, device)
                mask = torch.ones((1, 10), dtype=torch.bool, device=device)
                if stage in [STAGE_FLIP_1, STAGE_FLIP_2]:
                    for i in range(9):
                        mask[0, i] = bool(game.visible[curr_p][i])
                    mask[0, 9] = True
                else:
                    mask[0, :] = False

                with torch.no_grad():
                    action_idx, _, _, _ = agent.get_action(obs_tensor, action_masks=mask, deterministic=True)
                    action = action_idx.item()

                _, _, _, msg = game.step(action)
                print(f"Player {curr_p}: {msg}")
                time.sleep(0.5)


if __name__ == "__main__":
    play()
