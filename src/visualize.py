import time
import random
from .golf_game import GolfGame, STAGE_ARRANGE, STAGE_FLIP_1, STAGE_FLIP_2

def run_random_game():
    game = GolfGame()
    
    while True:
        game.render()
        time.sleep(0.5) 
        
        curr_player = game.current_turn_idx
        stage = game.player_stages[curr_player]
        
        action = 0
        
        if stage == STAGE_ARRANGE:
            action = 0
        elif stage == STAGE_FLIP_1 or stage == STAGE_FLIP_2:
            hidden_indices = [i for i in range(9) if not game.visible[curr_player][i]]
            if hidden_indices:
                action = random.choice(hidden_indices)
        else:
            action = random.randint(0, 9)
            
        # Step returns info, which contains our formatted log string
        _, _, _, info_msg = game.step(action)
        
        print(f"Player {curr_player} Action: {info_msg}")

if __name__ == "__main__":
    run_random_game()