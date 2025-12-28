import random
import numpy as np
from colorama import Fore, Style, init
from .consts import *

init(autoreset=True)

class GolfGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.players = list(range(NUM_PLAYERS))
        self.dealer_idx = random.randint(0, NUM_PLAYERS - 1)
        self.current_turn_idx = (self.dealer_idx + 1) % NUM_PLAYERS
        
        self.deck = list(range(TOTAL_CARDS))
        random.shuffle(self.deck)

        self.hands = np.zeros((NUM_PLAYERS, CARDS_PER_PLAYER), dtype=int)
        self.visible = np.zeros((NUM_PLAYERS, CARDS_PER_PLAYER), dtype=bool)
        
        for i in range(CARDS_PER_PLAYER):
            for p in range(NUM_PLAYERS):
                self.hands[p][i] = self.deck.pop()

        self.discard_pile = [self.deck.pop()]
        self.draw_pile = self.deck
        
        self.scores = np.zeros(NUM_PLAYERS, dtype=float)
        
        # Track scores for the current round as they are locked in
        self.projected_round_scores = np.zeros(NUM_PLAYERS, dtype=int)
        
        self.player_stages = [STAGE_ARRANGE] * NUM_PLAYERS
        
        self.finisher_idx = -1 
        self.final_turns_map = [False] * NUM_PLAYERS 

        return self._get_observation()

    def _get_card_color(self, card_id):
        return 1 if card_id >= 52 else 0

    def _reshuffle_discard_if_needed(self):
        if len(self.draw_pile) == 0:
            if len(self.discard_pile) <= 1:
                return 
            top_card = self.discard_pile.pop()
            self.draw_pile = self.discard_pile
            random.shuffle(self.draw_pile)
            self.discard_pile = [top_card]

    # --- ACTION HANDLER ---
    def step(self, action_idx):
        p_idx = self.current_turn_idx
        stage = self.player_stages[p_idx]
        log_msg = "" 

        # 1. ARRANGEMENT PHASE
        if stage == STAGE_ARRANGE:
            np.random.shuffle(self.hands[p_idx])
            self.player_stages[p_idx] = STAGE_FLIP_1
            return self._get_observation(), 0, False, "Arranged Hand"

        # 2. FLIP 1
        elif stage == STAGE_FLIP_1:
            if 0 <= action_idx < 9 and not self.visible[p_idx][action_idx]:
                old_repr = self._render_card(self.hands[p_idx][action_idx], visible=False)
                self.visible[p_idx][action_idx] = True
                new_repr = self._render_card(self.hands[p_idx][action_idx], visible=True)
                log_msg = f"Flip 1: Slot {action_idx} {old_repr} -> {new_repr}"
                self.player_stages[p_idx] = STAGE_FLIP_2
            return self._get_observation(), 0, False, log_msg

        # 3. FLIP 2
        elif stage == STAGE_FLIP_2:
            if 0 <= action_idx < 9 and not self.visible[p_idx][action_idx]:
                old_repr = self._render_card(self.hands[p_idx][action_idx], visible=False)
                self.visible[p_idx][action_idx] = True
                new_repr = self._render_card(self.hands[p_idx][action_idx], visible=True)
                log_msg = f"Flip 2: Slot {action_idx} {old_repr} -> {new_repr}"
                self._advance_init_turn()
            return self._get_observation(), 0, False, log_msg

        # 4. PLAY PHASE 1 (Decide on Top Discard)
        elif stage == STAGE_PLAY_DRAW:
            if action_idx == 9:
                self._reshuffle_discard_if_needed()
                new_card = self.draw_pile.pop()
                self.discard_pile.append(new_card)
                drawn_repr = self._render_card(new_card, visible=True)
                log_msg = f"Passed Discard. Drew {drawn_repr}"
                self.player_stages[p_idx] = STAGE_PLAY_DISCARD
            elif 0 <= action_idx < 9:
                top_discard_val = self.discard_pile[-1]
                old_val = self.hands[p_idx][action_idx]
                was_visible = self.visible[p_idx][action_idx]
                
                # RENDER LOGIC UPDATE
                card_in_hand_repr = self._render_card(old_val, was_visible)
                if not was_visible:
                    # Reveal what it was
                    real_val = self._render_card(old_val, visible=True)
                    card_in_hand_repr += f" (was {real_val})"
                
                new_card_repr = self._render_card(top_discard_val, visible=True)
                
                self._swap_card(p_idx, action_idx)
                log_msg = f"Took Discard: Slot {action_idx} {card_in_hand_repr} -> {new_card_repr}"
                self._end_turn(p_idx)
            return self._get_observation(), 0, False, log_msg

        # 5. PLAY PHASE 2 (Decide on New Top Discard)
        elif stage == STAGE_PLAY_DISCARD:
            if action_idx == 9:
                log_msg = "Passed Draw Card"
                self._end_turn(p_idx)
            elif 0 <= action_idx < 9:
                top_discard_val = self.discard_pile[-1] 
                old_val = self.hands[p_idx][action_idx]
                was_visible = self.visible[p_idx][action_idx]

                # RENDER LOGIC UPDATE
                card_in_hand_repr = self._render_card(old_val, was_visible)
                if not was_visible:
                    real_val = self._render_card(old_val, visible=True)
                    card_in_hand_repr += f" (was {real_val})"

                new_card_repr = self._render_card(top_discard_val, visible=True)
                
                self._swap_card(p_idx, action_idx)
                log_msg = f"Kept Draw: Slot {action_idx} {card_in_hand_repr} -> {new_card_repr}"
                self._end_turn(p_idx)
            return self._get_observation(), 0, False, log_msg
            
        return self._get_observation(), 0, False, "Invalid State"

    def _swap_card(self, p_idx, slot_idx):
        new_card = self.discard_pile.pop()
        old_card = self.hands[p_idx][slot_idx]
        self.hands[p_idx][slot_idx] = new_card
        self.visible[p_idx][slot_idx] = True
        self.discard_pile.append(old_card)

    def _advance_init_turn(self):
        next_p = (self.current_turn_idx + 1) % NUM_PLAYERS
        if self.player_stages[next_p] == STAGE_ARRANGE:
            self.current_turn_idx = next_p
        else:
            self.current_turn_idx = (self.dealer_idx + 1) % NUM_PLAYERS
            for i in range(NUM_PLAYERS):
                self.player_stages[i] = STAGE_PLAY_DRAW

    def _calculate_single_score(self, p_idx):
        """Calculates score for a single player based on current visible cards."""
        hand = self.hands[p_idx]
        cols = [[0,3,6], [1,4,7], [2,5,8]]
        p_score = 0
        for col_indices in cols: 
            idx1, idx2, idx3 = col_indices
            val1, val2, val3 = hand[idx1], hand[idx2], hand[idx3]
            face1, face2, face3 = val1%13, val2%13, val3%13
            
            if face1 == face2 == face3:
                col_score = 0
            else:
                col_score = POINT_VALUES[face1] + POINT_VALUES[face2] + POINT_VALUES[face3]
            p_score += col_score
        return p_score

    def _end_turn(self, p_idx):
        # 1. Did this move trigger the end game?
        if np.all(self.visible[p_idx]):
            if self.finisher_idx == -1:
                self.finisher_idx = p_idx 
                self.visible[p_idx][:] = True
                self.projected_round_scores[p_idx] = self._calculate_single_score(p_idx)
        
        # 2. Was this a final turn in an already triggered end game?
        if self.finisher_idx != -1:
            if p_idx != self.finisher_idx:
                self.final_turns_map[p_idx] = True
                self.visible[p_idx][:] = True
                self.projected_round_scores[p_idx] = self._calculate_single_score(p_idx)

            # Check if everyone is done
            completed_players = 0
            for i in range(NUM_PLAYERS):
                if i == self.finisher_idx: continue
                if self.final_turns_map[i]:
                    completed_players += 1
            if completed_players >= (NUM_PLAYERS - 1):
                self._calculate_scores_and_reset()
                return

        # 3. Advance Turn
        self.current_turn_idx = (self.current_turn_idx + 1) % NUM_PLAYERS
        
        if self.finisher_idx != -1 and self.current_turn_idx == self.finisher_idx:
             self._calculate_scores_and_reset()
             return

        self.player_stages[self.current_turn_idx] = STAGE_PLAY_DRAW

    def _calculate_scores_and_reset(self):
        # 1. Reveal EVERYTHING
        self.visible[:] = True
        
        # 2. RENDER THE FINAL STATE BEFORE WIPING
        print(f"\n{Fore.MAGENTA}" + "="*15 + " FINAL REVEAL " + "="*15 + f"{Style.RESET_ALL}")
        self.render()
        
        # 3. Calculate Scores
        round_points = np.zeros(NUM_PLAYERS, dtype=int)
        for p in range(NUM_PLAYERS):
            round_points[p] = self._calculate_single_score(p)
        
        print(f"\n{Fore.GREEN}--- ROUND OVER ---{Style.RESET_ALL}")
        print(f"Points Added: {round_points}")
        self.scores += round_points
        
        if np.any(self.scores >= 100):
            print(f"\n{Fore.RED}GAME OVER{Style.RESET_ALL}")
            print(f"Final Scores: {self.scores}")
            # Find winner
            winner = np.argmin(self.scores)
            print(f"WINNER: Player {winner}")
            exit() 
        else:
            print(f"Current Totals: {self.scores}")
            input(f"{Fore.CYAN}Press Enter to start next round...{Style.RESET_ALL}")
            self._reset_round_state()

    def _reset_round_state(self):
        self.deck = list(range(TOTAL_CARDS))
        random.shuffle(self.deck)
        self.hands = np.zeros((NUM_PLAYERS, CARDS_PER_PLAYER), dtype=int)
        self.visible = np.zeros((NUM_PLAYERS, CARDS_PER_PLAYER), dtype=bool)
        for i in range(CARDS_PER_PLAYER):
            for p in range(NUM_PLAYERS):
                self.hands[p][i] = self.deck.pop()
        self.discard_pile = [self.deck.pop()]
        self.draw_pile = self.deck
        
        # Reset turn order
        self.dealer_idx = (self.dealer_idx + 1) % NUM_PLAYERS
        self.current_turn_idx = (self.dealer_idx + 1) % NUM_PLAYERS
        
        # Reset State
        self.player_stages = [STAGE_ARRANGE] * NUM_PLAYERS
        self.finisher_idx = -1
        self.final_turns_map = [False] * NUM_PLAYERS
        self.projected_round_scores = np.zeros(NUM_PLAYERS, dtype=int)

    def _get_observation(self):
        return np.zeros(OBS_SIZE)

    # --- RENDERER ---
    def render(self):
        print("\n" + "="*40)
        print(f"Dealer: Player {self.dealer_idx}")
        print(f"Current Turn: Player {self.current_turn_idx}")
        
        if self.finisher_idx != -1:
            print(f"{Fore.YELLOW}*** FINAL TURNS TRIGGERED BY PLAYER {self.finisher_idx} ***{Style.RESET_ALL}")
            
        print(f"Stage: {self.player_stages[self.current_turn_idx]}")
        
        if len(self.discard_pile) > 0:
            print(f"Top Discard: {self._render_card(self.discard_pile[-1], visible=True)}")
        else:
            print("Top Discard: [Empty]")
            
        if len(self.draw_pile) > 0:
            draw_top = self.draw_pile[-1]
            color_str = "Blue" if draw_top >= 52 else "Red"
            color_code = Fore.BLUE if draw_top >= 52 else Fore.RED
            print(f"Top Draw Pile: {color_code}[ {color_str} Back ]{Style.RESET_ALL}")
        else:
            print("Top Draw Pile: [Empty]")
            
        print("-" * 20)
        
        for p in range(NUM_PLAYERS):
            prefix = ">> " if p == self.current_turn_idx else "   "
            
            score_str = f"{self.scores[p]}"
            
            is_finished = (p == self.finisher_idx) or self.final_turns_map[p]
            # Show projected score if revealed, but if visible[:] is all true (end of game), show calc
            if is_finished or np.all(self.visible[p]):
                round_s = self._calculate_single_score(p)
                total_s = self.scores[p] + round_s
                score_str = f"{self.scores[p]} + {round_s} = {total_s}"
            
            print(f"{prefix}Player {p} (Score: {score_str})")
            
            if self.player_stages[p] == STAGE_ARRANGE:
                r_count = np.sum(self.hands[p] < 52)
                b_count = 9 - r_count
                r_txt = f"{Fore.RED}{r_count} Red{Style.RESET_ALL}"
                b_txt = f"{Fore.BLUE}{b_count} Blue{Style.RESET_ALL}"
                print(f"   [Pending Arrangement: {r_txt}, {b_txt}]")
                print("")
                continue 
            
            for row in range(3):
                row_str = "   "
                for col in range(3):
                    idx = row * 3 + col
                    card_val = self.hands[p][idx]
                    is_vis = self.visible[p][idx]
                    row_str += self._render_card(card_val, is_vis) + " "
                print(row_str)
            print("")

    def _render_card(self, card_id, visible=True):
        is_blue = card_id >= 52
        color_code = Fore.BLUE if is_blue else Fore.RED
        
        if visible:
            face = card_id % 13
            face_str = FACE_STR_MAP[face]
            return f"{color_code}[ {face_str} ]{Style.RESET_ALL}"
        else:
            return f"{color_code}[ ? ]{Style.RESET_ALL}"