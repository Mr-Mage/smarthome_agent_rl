"""Paired success inference with exact McNemar, stratified bootstrap and Holm correction."""
import math
import random


def mcnemar(baseline, variant):
    if len(baseline) != len(variant) or not baseline:
        raise ValueError('Paired nonempty outcomes required')
    wins = sum(not b and v for b, v in zip(baseline, variant))
    losses = sum(b and not v for b, v in zip(baseline, variant))
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) /
            2 ** discordant) if discordant else 1.0
    return {'wins': wins, 'losses': losses, 'p_exact': p,
            'success_delta': (wins - losses) / len(baseline)}


def bootstrap_ci(strata, *, repeats=5000, seed=20261004):
    if not strata or any(not values for values in strata):
        raise ValueError('Nonempty strata required')
    rng = random.Random(seed)
    total = sum(len(values) for values in strata)
    draws = sorted(sum(sum(rng.choice(values) for _ in values) for values in strata) / total
                   for _ in range(repeats))
    return [draws[int(0.025 * repeats)], draws[min(repeats - 1, int(0.975 * repeats))]]


def holm(p_values):
    adjusted, previous = {}, 0.0
    ordered = sorted(p_values, key=lambda key: p_values[key])
    for i, key in enumerate(ordered):
        previous = min(1.0, max(previous, (len(ordered) - i) * p_values[key]))
        adjusted[key] = previous
    return adjusted
