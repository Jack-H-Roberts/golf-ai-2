# Golf — Rules of the Game

A 5-player card game played with two standard decks. The lowest total score wins.

---

## Setup

- **5 players**, **2 standard 52-card decks**, no jokers (104 cards total).
- The two decks are visually distinguishable (one "red" deck, one "blue" deck). All cards' **colors are visible at all times** to all players, even when face-down. This is the only information leak from the deck identity — face values are still hidden until a card is flipped.
- **9 cards dealt face-down** to each player, arranged into a 3×3 grid (45 cards used).
- **1 card revealed face-up** to start the discard pile.
- **Top of the draw pile has its color shown** to all players (face still hidden).

After dealing: 45 cards in hands, 1 on top of discard, 1 on top of draw pile (color shown), 57 face-down in the rest of the draw pile.

## Card Values

| Card | Value |
|------|-------|
| A    | 1     |
| 2    | 2     |
| 3    | 3     |
| 4    | 4     |
| 5    | 5     |
| 6    | 6     |
| 7    | 7     |
| **8**    | **−2**    |
| 9    | 9     |
| 10   | 10    |
| J    | 10    |
| Q    | 10    |
| K    | 0     |

8s are the "good" cards — they actively lower your score. Kings are neutral.

---

## Setup Phase (each player, in dealer order)

Before the play phase begins, every player goes through three steps in sequence. Players act in dealer order (player after dealer goes first).

### 1. ARRANGE

Place your 9 face-down cards into a 3×3 grid. You can rearrange your cards based on the colors you see. Cards' positions are **fixed once committed** — you can't reshuffle later. Faces remain hidden.

### 2. FLIP 1

Choose any one of your 9 slots and flip that card face-up. Its face is now visible to all players.

### 3. FLIP 2

Choose any other slot (not the one you just flipped) and flip it face-up. Its face is now visible to all players.

After all 5 players have completed setup, each player has:
- A committed 3×3 grid.
- 2 cards face-up (visible).
- 7 cards face-down (color visible, face hidden).

The play phase begins, again in dealer order.

---

## Play Phase

On your turn:

### First choice: see the top discard

The top of the discard pile is face-up — both color and face are known. Decide:

- **TAKE**: Place the top discard into one of your 9 slots. Whatever was in that slot becomes the new top of discard.
  - If the slot was face-down, the card you replaced now has its face revealed (it goes face-up onto the discard).
  - If the slot was face-up, you're just swapping the visible cards.
  - The slot you placed into is now face-up.
- **PASS**: The top discard is set aside (out of play permanently). The top of the draw pile is now revealed and becomes the new top of discard. The color of the new (next) top draw pile card is also revealed.

### Second choice (only if you passed)

You're now looking at a face-up card on top of the discard — it's the card you just drew from the deck.

- **TAKE**: Same as above. Place into a slot, replaced card becomes the new top of discard.
- **PASS**: The drawn card stays as the top of discard. Your turn ends.

The net effect of pass-pass: the original top discard is gone (out of play); the drawn card is now on top of the discard pile.

---

## End of Game

The game ends when EITHER condition triggers:

### Hand completion (standard rule)

When a player's action makes all 9 of their slots face-up, that player becomes the **finisher**. Each other player gets exactly one more turn (their "final turn"), continuing in order from the finisher. After the last final turn, the game ends.

If a player took their final turn without filling their hand, **all of their remaining face-down cards are revealed and scored as-is** — no more chances to swap them out.

### Draw pile exhausted

If a player would pass during their first choice but the draw pile is empty, the game ends immediately. All face-down cards are revealed and scored as-is, no more turns played.

---

## Scoring

When the game ends, every player's score is computed:

For each of the 3 columns (slots [0,3,6], [1,4,7], [2,5,8] vertically):

- If all 3 cards have the **same face** (regardless of color — three 7s in any color combination): the column scores **0**.
- Otherwise: the column scores the **sum of point values** of the 3 cards.

Total score = sum of the 3 column scores.

**The player with the lowest total wins.** Ties are possible.

---

## Strategic Notes

- **3-of-a-kind columns are powerful**: any column of 3 same-face cards scores 0, regardless of the cards' values. Even three Queens (worth 10 each individually) score 0 as a column.
- **8s should not be in 3-of-a-kind columns**: if you have three 8s in a column, the rule says the column scores 0. You'd score −6 if it were just summed.
- **Triggering the end with a bad hand is a losing move**: if you fill your last face-down slot but you're still scoring high, you've capped your downside but signaled the game to end before you could improve. Every other player gets one more turn to swap their bad cards out. This is a real strategic decision.
- **Information asymmetry from face-up cards**: each turn, you can see what others have flipped. They might be building a column you can disrupt by holding a key card.

---

## Quick Reference

- 5 players × 9 cards × 3×3 grid
- 2 decks (red + blue), 104 cards total
- Setup: ARRANGE → FLIP1 → FLIP2 (each player, dealer order)
- Play: TAKE / PASS-and-TAKE / PASS-PASS (each turn)
- End: hand-complete + final round, OR deck-empty
- Score: column 3-of-a-kind = 0, else sum; lowest wins
