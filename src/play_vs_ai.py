import torch
import numpy as np
import time
from colorama import Fore, Style, init

from src.golf_game import GolfGame, STAGE_ARRANGE, STAGE_FLIP_1, STAGE_FLIP_2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD
from src.model import GolfNet
from src.consts import *

init(autoreset=True)

MODEL_PATH = "latest_model.pt"

# --- ADAPTER: CONVERT GAME STATE TO TENSOR ---
def build_single_observation(game, p_idx, device):
    """
    Manually constructs the 741-float input vector for a single player
    from the GolfGame class state.
    """
    obs_list = []
    
    # 1. Stage Flags (5 bits)
    stage = game.player_stages[p_idx]
    st_oh = np.zeros(5)
    st_oh[stage] = 1.0
    obs_list.extend(st_oh)
    
    # 2. Positions (5 bits) - My Seat Index
    pos_oh = np.zeros(NUM_PLAYERS)
    pos_oh[p_idx] = 1.0
    obs_list.extend(pos_oh)
    
    # --- ROTATION INDICES ---
    rot_idxs = [(p_idx + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
    
    # 3. Cumulative Scores (5 floats)
    scores = [game.scores[i] / 100.0 for i in rot_idxs]
    obs_list.extend(scores)
    
    # 4. Counts (Red Cards) (5 floats)
    counts = []
    for i in rot_idxs:
        r_count = np.sum(game.hands[i] < 52)
        counts.append(r_count)
    obs_list.extend(counts)
    
    # 5. Finisher Trigger (5 bits)
    fin_oh = np.zeros(5)
    if game.finisher_idx != -1:
        rel_fin = (game.finisher_idx - p_idx + NUM_PLAYERS) % NUM_PLAYERS
        fin_oh[rel_fin] = 1.0
    obs_list.extend(fin_oh)
    
    # 6. Top Discard (14 bits)
    if len(game.discard_pile) > 0:
        td = game.discard_pile[-1]
        is_blue = 1.0 if td >= 52 else 0.0
        face = td % 13
        face_oh = np.zeros(13)
        face_oh[face] = 1.0
    else:
        is_blue = 0.0
        face_oh = np.zeros(13)
    
    obs_list.append(is_blue)
    obs_list.extend(face_oh)
    
    # 7. Draw Pile Color (1 bit)
    if len(game.draw_pile) > 0:
        draw_top = game.draw_pile[-1]
        is_d_blue = 1.0 if draw_top >= 52 else 0.0
    else:
        is_d_blue = 0.0
    obs_list.append(is_d_blue)
    
    # 8. Graveyard Counts (26 ints)
    graveyard = np.zeros(26)
    dead_cards = game.discard_pile[:-1]
    for c in dead_cards:
        is_b = (c >= 52)
        face = c % 13
        idx = (13 if is_b else 0) + face
        graveyard[idx] += 1.0
    obs_list.extend(graveyard)
    
    # 9. Known Hands (675 bits)
    for i in rot_idxs:
        hand = game.hands[i]
        vis = game.visible[i]
        
        for j in range(9):
            card_vec = np.zeros(15)
            if vis[j]:
                c_val = hand[j]
                is_b = c_val >= 52
                face = c_val % 13
                card_vec[1 if is_b else 0] = 1.0
                card_vec[2 + face] = 1.0
            obs_list.extend(card_vec)
            
    return torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0).to(device)

# --- HUMAN INTERACTION ---
def human_arrange_phase(game):
    """
    Simplified Arrangement: User enters indices for Red cards.
    Blue cards automatically fill the remaining spots.
    """
    p_idx = 0
    hand = game.hands[p_idx]
    
    # Separate current hand into piles
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
            # Parse indices
            indices = list(map(int, choice.split()))
            
            # Validation
            if len(indices) != num_reds:
                print(f"{Fore.RED}Error: You entered {len(indices)} indices, but you have {num_reds} Red cards.{Style.RESET_ALL}")
                continue
                
            if any(i < 0 or i > 8 for i in indices):
                print(f"{Fore.RED}Error: Indices must be between 0 and 8.{Style.RESET_ALL}")
                continue
                
            if len(set(indices)) != len(indices):
                print(f"{Fore.RED}Error: Duplicate indices found.{Style.RESET_ALL}")
                continue
                
            # Construct New Hand
            new_hand = np.zeros(9, dtype=int)
            
            # Place Reds
            for i, r_val in zip(indices, red_cards):
                new_hand[i] = r_val
                
            # Place Blues in remaining slots
            remaining_slots = [s for s in range(9) if s not in indices]
            for i, b_val in zip(remaining_slots, blue_cards):
                new_hand[i] = b_val
                
            # Update Game State
            game.hands[p_idx] = new_hand
            print(f"{Fore.GREEN}Arrangement Accepted.{Style.RESET_ALL}")
            
            # Advance Stage
            game.player_stages[p_idx] = STAGE_FLIP_1
            break
            
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter numbers separated by spaces.{Style.RESET_ALL}")

def get_human_move(game):
    stage = game.player_stages[0]
    
    valid_moves = []
    if stage in [STAGE_FLIP_1, STAGE_FLIP_2]:
        valid_moves = [i for i in range(9) if not game.visible[0][i]]
    elif stage in [STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD]:
        valid_moves = list(range(10)) 
    
    print(f"{Fore.GREEN}Your Turn! Valid Moves: {valid_moves}{Style.RESET_ALL}")
    
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
    print("Loading AI Brain...")
    
    agent = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        agent.eval()
        print("Model Loaded!")
    except FileNotFoundError:
        print("Model not found. Run training first.")
        return

    game = GolfGame()
    
    print("Welcome to Golf AI. You are Player 0.")
    time.sleep(1)
    
    while True:
        curr_p = game.current_turn_idx
        stage = game.player_stages[curr_p]
        
        # --- HUMAN TURN ---
        if curr_p == 0:
            if stage == STAGE_ARRANGE:
                human_arrange_phase(game)
                continue 
            
            game.render()
            action = get_human_move(game)
            _, _, _, msg = game.step(action)
            print(f"You: {msg}")
            time.sleep(1)
            
        # --- AI TURN ---
        else:
            if stage == STAGE_ARRANGE:
                # AI randomizes arrangement (simulation of choice)
                _, _, _, msg = game.step(0)
            else:
                # 1. Build Observation
                obs_tensor = build_single_observation(game, curr_p, device)
                
                # 2. Get Mask
                mask = torch.ones((1, 10), dtype=torch.bool, device=device)
                if stage in [STAGE_FLIP_1, STAGE_FLIP_2]:
                    for i in range(9):
                        if game.visible[curr_p][i]: mask[0, i] = True
                        else: mask[0, i] = False 
                    mask[0, 9] = True 
                else:
                    mask[0, :] = False 
                
                # 3. Query Model
                with torch.no_grad():
                    action_idx, _, _, _ = agent.get_action(obs_tensor, action_masks=mask, deterministic=True)
                    action = action_idx.item()
                
                # 4. Step
                _, _, _, msg = game.step(action)
                print(f"Player {curr_p}: {msg}")
                time.sleep(0.5)

if __name__ == "__main__":
    play()