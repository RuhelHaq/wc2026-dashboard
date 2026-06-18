import numpy as np
from scipy.stats import poisson

def scoreline_matrix(lam_home, lam_away, max_goals=8, rho=-0.1):
    home_probs = np.array([poisson.pmf(i, lam_home) for i in range(max_goals + 1)])
    away_probs = np.array([poisson.pmf(j, lam_away) for j in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)

    # Dixon-Coles correction
    matrix[0, 0] *= 1 - lam_home * lam_away * rho
    matrix[1, 0] *= 1 + lam_away * rho
    matrix[0, 1] *= 1 + lam_home * rho
    matrix[1, 1] *= 1 - rho

    matrix /= matrix.sum()
    return matrix


def match_probabilities(lam_home, lam_away, max_goals=8, rho=-0.1):
    mat = scoreline_matrix(lam_home, lam_away, max_goals, rho)

    home_win = float(np.sum(np.tril(mat, -1)))
    draw     = float(np.sum(np.diag(mat)))
    away_win = float(np.sum(np.triu(mat, 1)))

    flat = [(mat[i, j], i, j) for i in range(max_goals + 1)
                               for j in range(max_goals + 1)]
    top5 = sorted(flat, reverse=True)[:5]

    return {
        'home_win':      round(home_win, 4),
        'draw':          round(draw, 4),
        'away_win':      round(away_win, 4),
        'top_scorelines': top5
    }
