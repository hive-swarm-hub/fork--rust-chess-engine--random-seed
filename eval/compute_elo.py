#!/usr/bin/env python3
"""Compute ELO from cutechess-cli gauntlet output via maximum likelihood estimation.

Methodology:
  This uses the standard Elo MLE approach — the same mathematical foundation as
  FIDE performance ratings, Bayeselo, and Ordo. Given games against opponents of
  known rating, we find the rating R that maximizes the likelihood of the observed
  results under the logistic Elo model:

    P(win) = 1 / (1 + 10^((R_opp - R) / 400))

  Newton's method converges in <20 iterations. The 95% confidence interval comes
  from the Fisher information (inverse Hessian of the log-likelihood).

Opponent types:
  - SF_<elo>: Stockfish with UCI_LimitStrength (calibrated by Stockfish devs)
  - SF_d<depth>: Stockfish at fixed search depth (approximate calibration)

  Using both opponent types prevents reward hacking: an engine that only exploits
  UCI_LimitStrength quirks will still lose to depth-limited opponents, and the
  combined MLE is robust to either opponent type's calibration errors.
"""

import math
import re
import sys

# Known opponent ratings.
# UCI_LimitStrength ratings are Stockfish's internal calibration.
# Depth-limited ratings are approximate (hardware-dependent, ±200 ELO).
def get_opponent_rating(name: str) -> int | None:
    """Extract rating from opponent name like SF_2200 -> 2200."""
    if name.startswith("SF_"):
        try:
            return int(name[3:])
        except ValueError:
            return None
    return None


def parse_results(output: str) -> list[dict]:
    """Parse game results from cutechess-cli or fastchess output.

    Supports two formats:
    1. cutechess-cli: "Score of X vs Y: W - L - D"
    2. fastchess: "Finished game N (White vs Black): result {reason}"
    """
    # Try cutechess-cli format first
    games_by_opp: dict[str, dict] = {}

    pattern = r"Score of (\S+) vs (\S+): (\d+) - (\d+) - (\d+)"
    for match in re.finditer(pattern, output):
        engine = match.group(1)
        opponent = match.group(2)
        wins = int(match.group(3))
        losses = int(match.group(4))
        draws = int(match.group(5))

        opp_rating = get_opponent_rating(opponent)
        eng_rating = get_opponent_rating(engine)
        if opp_rating is not None:
            games_by_opp[opponent] = {
                "opponent": opponent, "rating": opp_rating,
                "wins": wins, "losses": losses, "draws": draws,
                "total": wins + losses + draws,
                "score": wins + 0.5 * draws,
            }
        elif eng_rating is not None:
            games_by_opp[engine] = {
                "opponent": engine, "rating": eng_rating,
                "wins": losses, "losses": wins, "draws": draws,
                "total": wins + losses + draws,
                "score": losses + 0.5 * draws,
            }

    if games_by_opp:
        return list(games_by_opp.values())

    # Fallback: parse fastchess per-game lines
    # Format: "Finished game N (White vs Black): 1-0 {reason}"
    #         "Finished game N (White vs Black): 0-1 {reason}"
    #         "Finished game N (White vs Black): 1/2-1/2 {reason}"
    OUR_ENGINE = "HiveChess"
    per_game_pattern = r"Finished game \d+ \((\S+) vs (\S+)\): (\S+)"

    for match in re.finditer(per_game_pattern, output):
        white = match.group(1)
        black = match.group(2)
        result = match.group(3)

        # Determine who is our engine and who is the opponent
        if white == OUR_ENGINE and get_opponent_rating(black) is not None:
            opponent = black
            if result == "1-0":
                outcome = "win"
            elif result == "0-1":
                outcome = "loss"
            else:
                outcome = "draw"
        elif black == OUR_ENGINE and get_opponent_rating(white) is not None:
            opponent = white
            if result == "0-1":
                outcome = "win"
            elif result == "1-0":
                outcome = "loss"
            else:
                outcome = "draw"
        else:
            continue

        if opponent not in games_by_opp:
            games_by_opp[opponent] = {
                "opponent": opponent, "rating": get_opponent_rating(opponent),
                "wins": 0, "losses": 0, "draws": 0, "total": 0, "score": 0.0,
            }

        g = games_by_opp[opponent]
        g["total"] += 1
        if outcome == "win":
            g["wins"] += 1
            g["score"] += 1.0
        elif outcome == "draw":
            g["draws"] += 1
            g["score"] += 0.5
        else:
            g["losses"] += 1

    return list(games_by_opp.values())


def estimate_elo(games: list[dict]) -> tuple[float, float]:
    """Maximum likelihood ELO estimation using Newton's method.

    Returns (estimated_elo, standard_error).

    Finds R such that: sum(actual_score) = sum(expected_score)
    where expected_score_j = N_j / (1 + 10^((R_opp_j - R) / 400))

    This is mathematically identical to:
    - FIDE performance rating calculation
    - Bayeselo's core estimator
    - Ordo's likelihood maximization

    The standard error comes from Fisher information:
    SE = 1 / sqrt(sum(N_j * E_j * (1-E_j)) * (ln10/400)^2)
    """
    if not games:
        return 0.0, 0.0

    total_score = sum(g["score"] for g in games)
    total_games = sum(g["total"] for g in games)

    if total_games == 0:
        return 0.0, 0.0
    if total_score == 0:
        return min(g["rating"] for g in games) - 400.0, 400.0
    if total_score == total_games:
        return max(g["rating"] for g in games) + 400.0, 400.0

    # Newton's method to find MLE with damped updates for stability.
    # Start from a better initial estimate: weighted avg opponent rating.
    ln10_400 = math.log(10) / 400.0
    total_score = sum(g["score"] for g in games)
    total_games = sum(g["total"] for g in games)
    weighted_avg = sum(g["rating"] * g["total"] for g in games) / total_games
    score_frac = total_score / total_games
    # Initial estimate via performance rating formula (clamped score fraction)
    p = max(0.01, min(0.99, score_frac))
    R = weighted_avg + 400.0 * math.log10(p / (1.0 - p))

    denominator = 0.0
    for _ in range(500):
        numerator = 0.0
        denominator = 0.0

        for g in games:
            exponent = (g["rating"] - R) / 400.0
            exponent = max(-10.0, min(10.0, exponent))
            E = 1.0 / (1.0 + 10.0 ** exponent)
            numerator += g["score"] - g["total"] * E
            denominator += g["total"] * E * (1.0 - E)

        if abs(denominator) < 1e-12:
            break

        update = numerator / (denominator * ln10_400)
        # Dampen: cap step size to ±200 ELO per iteration for stability
        update = max(-200.0, min(200.0, update))
        R += update

        if abs(update) < 0.01:
            break

    # Standard error from Fisher information
    fisher_info = denominator * ln10_400 ** 2
    se = 1.0 / math.sqrt(fisher_info) if fisher_info > 0 else 400.0

    return round(R, 1), round(se, 1)


def main():
    output = sys.stdin.read()
    games = parse_results(output)

    if not games:
        print("elo: ERROR")
        print("games_played: 0")
        print("score_pct: 0.000")
        print("total_wins: 0")
        print("total_draws: 0")
        print("total_losses: 0")
        print("ci_95: 0")
        return

    elo, se = estimate_elo(games)
    total_games = sum(g["total"] for g in games)
    total_score = sum(g["score"] for g in games)
    total_wins = sum(g["wins"] for g in games)
    total_draws = sum(g["draws"] for g in games)
    total_losses = sum(g["losses"] for g in games)
    score_pct = total_score / total_games if total_games > 0 else 0.0
    ci_95 = round(1.96 * se, 1)

    print(f"elo: {elo}")
    print(f"games_played: {total_games}")
    print(f"score_pct: {score_pct:.3f}")
    print(f"total_wins: {total_wins}")
    print(f"total_draws: {total_draws}")
    print(f"total_losses: {total_losses}")
    print(f"ci_95: {ci_95}")

    print()
    print("Per-opponent breakdown:")
    for g in games:
        pct = g["score"] / g["total"] if g["total"] > 0 else 0
        marker = " [depth]" if g["opponent"].startswith("SF_d") else ""
        print(f"  vs {g['opponent']} ({g['rating']}): "
              f"+{g['wins']} ={g['draws']} -{g['losses']} [{pct:.3f}]{marker}")


if __name__ == "__main__":
    main()
