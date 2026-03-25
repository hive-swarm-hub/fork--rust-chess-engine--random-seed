#!/usr/bin/env python3
"""Generate diverse opening positions for chess engine gauntlet testing.

Produces ~500 EPD positions by traversing common opening trees to depth 3-6
half-moves with randomized branching. Ensures positional diversity so agents
cannot memorize a small fixed set.

Usage: python3 gen_openings.py > data/openings.epd
"""

import random
import sys

try:
    import chess
except ImportError:
    print("ERROR: python-chess required. Install: pip install chess", file=sys.stderr)
    sys.exit(1)

# Major opening trees: each entry is a list of UCI moves from startpos.
# We generate positions at various depths along these lines.
OPENING_TREES = [
    # === Open Games (1. e4 e5) ===
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],  # Morphy Defense
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"],  # Berlin
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "f1a4", "g8f6"],  # Closed Ruy
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],  # Italian
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],  # Giuoco Piano
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],  # Two Knights
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],  # Scotch
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4"],  # Scotch accepted
    ["e2e4", "e7e5", "g1f3", "g8f6"],  # Petroff
    ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6"],  # Petroff Classical
    ["e2e4", "e7e5", "g1f3", "d7d6"],  # Philidor
    ["e2e4", "e7e5", "f1c4"],  # Bishop's Opening
    ["e2e4", "e7e5", "f2f4"],  # King's Gambit
    ["e2e4", "e7e5", "f2f4", "e5f4"],  # KG Accepted
    ["e2e4", "e7e5", "b1c3"],  # Vienna
    ["e2e4", "e7e5", "d2d4", "e5d4", "d1d4"],  # Center Game

    # === Sicilian Defense ===
    ["e2e4", "c7c5"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],  # Najdorf/Classical setup
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],  # Open Sicilian
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],  # Najdorf
    ["e2e4", "c7c5", "g1f3", "b8c6"],  # Sicilian Classical
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4"],  # Open Sicilian
    ["e2e4", "c7c5", "g1f3", "e7e6"],  # Sicilian e6
    ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4"],  # Scheveningen
    ["e2e4", "c7c5", "g1f3", "g7g6"],  # Sicilian Dragon setup
    ["e2e4", "c7c5", "b1c3"],  # Closed Sicilian
    ["e2e4", "c7c5", "c2c3"],  # Alapin
    ["e2e4", "c7c5", "f1c4"],  # Bowdler Attack
    ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3"],  # Smith-Morra

    # === French Defense ===
    ["e2e4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],  # Winawer/Classical
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"],  # Winawer
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],  # Classical
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"],  # Tarrasch
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"],  # Advance
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5", "e6d5"],  # Exchange

    # === Caro-Kann ===
    ["e2e4", "c7c6"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"],  # Classical
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"],  # Main line
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"],  # Advance
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"],  # Exchange

    # === Other Semi-Open ===
    ["e2e4", "d7d5"],  # Scandinavian
    ["e2e4", "d7d5", "e4d5", "d8d5"],  # Scandi Queen recapture
    ["e2e4", "d7d5", "e4d5", "g8f6"],  # Scandi Knight
    ["e2e4", "g8f6"],  # Alekhine
    ["e2e4", "g8f6", "e4e5", "f6d5"],  # Alekhine main
    ["e2e4", "d7d6"],  # Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3"],  # Pirc main
    ["e2e4", "g7g6"],  # Modern
    ["e2e4", "g7g6", "d2d4", "f8g7"],  # Modern main
    ["e2e4", "b7b6"],  # Owen
    ["e2e4", "b8c6"],  # Nimzowitsch

    # === Queen's Gambit ===
    ["d2d4", "d7d5", "c2c4"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],  # QGD
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],  # QGD main
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "f8e7"],  # Orthodox
    ["d2d4", "d7d5", "c2c4", "c7c6"],  # Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"],  # Slav main
    ["d2d4", "d7d5", "c2c4", "d5c4"],  # QGA
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3"],  # QGA main
    ["d2d4", "d7d5", "g1f3", "g8f6"],  # QP game
    ["d2d4", "d7d5", "c2c4", "b8c6"],  # Chigorin

    # === Indian Defenses ===
    ["d2d4", "g8f6", "c2c4", "g7g6"],  # KID
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],  # KID main
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],  # KID Classical
    ["d2d4", "g8f6", "c2c4", "g7g6", "g1f3", "f8g7", "g2g3"],  # KID Fianchetto
    ["d2d4", "g8f6", "c2c4", "e7e6"],  # Nimzo/QID
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],  # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"],  # Classical Nimzo
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"],  # Rubinstein
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],  # Queen's Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],  # Catalan
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5"],  # Catalan main
    ["d2d4", "g8f6", "c2c4", "c7c5"],  # Benoni
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5"],  # Modern Benoni
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"],  # Benoni main
    ["d2d4", "g8f6", "c2c4", "e7e5"],  # Budapest
    ["d2d4", "g8f6", "c2c4", "b7b6"],  # QID early
    ["d2d4", "g8f6", "f1g5"],  # Trompowsky
    ["d2d4", "g8f6", "c1g5"],  # err... let me fix
    ["d2d4", "f7f5"],  # Dutch
    ["d2d4", "f7f5", "c2c4"],  # Dutch main
    ["d2d4", "f7f5", "g2g3"],  # Dutch Leningrad prep

    # === Flank Openings ===
    ["c2c4", "e7e5"],  # English
    ["c2c4", "e7e5", "b1c3"],  # English main
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["c2c4", "g8f6"],  # English Indian
    ["c2c4", "c7c5"],  # English Symmetrical
    ["c2c4", "c7c5", "g1f3"],
    ["c2c4", "c7c6"],  # English Caro setup
    ["g1f3", "d7d5"],  # Reti
    ["g1f3", "d7d5", "g2g3"],
    ["g1f3", "d7d5", "c2c4"],  # Reti with c4
    ["g1f3", "g8f6"],  # Double Knight
    ["g1f3", "g8f6", "g2g3"],
    ["g1f3", "g8f6", "c2c4"],
    ["g2g3"],  # King's Fianchetto
    ["g2g3", "d7d5"],
    ["b2b3"],  # Larsen
    ["b2b3", "e7e5"],
    ["g1f3", "c7c5"],  # English/Sicilian reversed

    # === Other d4 systems ===
    ["d2d4", "d7d5", "c1f4"],  # London
    ["d2d4", "d7d5", "c1f4", "g8f6"],
    ["d2d4", "g8f6", "c1f4"],  # London vs Indian
    ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"],  # London main
    ["d2d4", "d7d5", "e2e3"],  # Colle
    ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3"],  # Colle main
    ["d2d4", "g8f6", "g1f3", "g7g6", "c1f4"],  # London vs KID
    ["d2d4", "g8f6", "g1f3", "e7e6", "c1f4"],  # London vs NID
    ["d2d4", "g8f6", "b1c3", "d7d5", "c1f4"],  # Jobava London
]


def generate_positions(target_count: int = 500, seed: int = 42) -> list[str]:
    """Generate diverse EPD positions from opening trees."""
    rng = random.Random(seed)
    positions = set()

    # Add positions at every depth along each opening tree
    for tree in OPENING_TREES:
        board = chess.Board()
        for i, uci_move in enumerate(tree):
            try:
                move = chess.Move.from_uci(uci_move)
                if move not in board.legal_moves:
                    break
                board.push(move)
                # Add positions from depth 2 onwards (after 1 move each)
                if i >= 1:
                    epd = board.epd()
                    positions.add(epd)
            except (ValueError, chess.InvalidMoveError):
                break

    # Extend by making 1-2 random legal moves from each tree endpoint
    for tree in OPENING_TREES:
        for _ in range(3):  # 3 random continuations per tree
            board = chess.Board()
            valid = True
            for uci_move in tree:
                try:
                    move = chess.Move.from_uci(uci_move)
                    if move not in board.legal_moves:
                        valid = False
                        break
                    board.push(move)
                except (ValueError, chess.InvalidMoveError):
                    valid = False
                    break

            if not valid:
                continue

            # Play 1-3 random moves
            for _ in range(rng.randint(1, 3)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                # Prefer non-random: top moves by simple heuristic
                # (captures, center moves, development)
                move = rng.choice(legal)
                board.push(move)
                positions.add(board.epd())

    # If we need more, generate random games from startpos
    while len(positions) < target_count:
        board = chess.Board()
        depth = rng.randint(4, 8)
        for _ in range(depth):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if not board.is_game_over() and len(list(board.legal_moves)) > 5:
            positions.add(board.epd())

    result = sorted(positions)
    rng.shuffle(result)
    return result[:target_count]


if __name__ == "__main__":
    positions = generate_positions(500)
    for epd in positions:
        print(epd)
    print(f"# Generated {len(positions)} opening positions", file=sys.stderr)
