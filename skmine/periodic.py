import numpy as np

import pandas as pd

from skmine.base import MDLOptimizer
from skmine.base import BaseMiner

log = np.log2


def window_stack(x, stepsize=1, width=3):
    """
    Returns
        np.array of shape (x.shape[0] - width + stepsize, width)
    """
    n = x.shape[0]
    subs = [x[i : 1 + n + i - width : stepsize] for i in range(0, width)]
    return np.vstack(subs).T


def residual_length(S_alpha, n_event_tot, delta_S):
    """
    compute L(o) = L(t) + L(a) for all (a, t) in S_alpha
    i.e the length from a block of residual events

    Parameters
    ----------
    S_alpha: np.narray of shape (|a|, ) or scalar
        array containing indices for events to consider

    n_event_tot: int
        number of events in the original events

    delta_S: int
        max - min from original events
    """
    if isinstance(S_alpha, np.ndarray):
        card = S_alpha.shape[0]
    else:  # single value as scalar
        card = 1
    return log(delta_S + 1) - log(card / float(n_event_tot))


def cycle_length(S_alpha, inter, n_event_tot, dS):
    """
    Parameters
    ----------
    S_alpha : np.array of type int64
        a collection of cycles, all having the same length : r
        The width of S is then r

    inter: np.array of type int64
        a collection of inter occurences, all having the same length: r - 1

    n_event_tot: int
        number of events in the original events

    dS: int
        max - min from original events

    Returns
    -------
    tuple()
    """
    r = S_alpha.shape[1]
    assert inter.shape[1] == r - 1  # check inter occurences compliant with events
    p = np.median(inter, axis=1)
    E = inter - p.reshape((-1, 1))
    dE = E.sum(axis=1)
    S_alpha_size = (
        len(S_alpha) + r - 1
    )  # beware of this if S_alpha was generated with stepsize > 1

    L_a = -log(S_alpha_size / n_event_tot)  # FIXME
    L_r = log(S_alpha_size)
    L_p = log(np.floor((dS - dE) / (r - 1)))
    L_tau = log(dS - dE - (r - 1) * p + 1)
    L_E = 2 * E.shape[1] + np.abs(E).sum(axis=1)

    return L_a, L_r, L_p, L_tau, L_E


def get_table_dyn(S_a: pd.DatetimeIndex, n_event_tot: int):
    S_a = S_a.astype("int64")
    diffs = np.diff(S_a)
    triples = window_stack(S_a, width=3)
    diff_pairs = window_stack(diffs, width=2)
    delta_S = S_a.max() - S_a.min()

    score_one = residual_length(1, n_event_tot, delta_S)

    L_a, L_r, L_p, L_tau, L_E = cycle_length(triples, diff_pairs, len(S_a), delta_S)
    triple_scores = L_a + L_r + L_p + L_tau + L_E
    change = triple_scores > 3 * score_one
    triple_scores[change] = 3 * score_one  # inplace replacement
    cut_points = np.array([-1] * len(triple_scores), dtype=object)
    cut_points[~change] = None

    scores = dict(zip(((i, i + 2) for i in range(len(triple_scores))), triple_scores))
    cut_points = dict(zip(scores.keys(), cut_points))

    for k in range(4, len(S_a) + 1):
        w = window_stack(S_a, width=k)
        _diffs = window_stack(diffs, width=k - 1)
        _s = sum(cycle_length(w, _diffs, len(S_a), delta_S))

        for ia, best_score in enumerate(_s):
            cut_point = None
            iz = ia + k - 1
            for im in range(ia, iz):
                if im - ia + 1 < 3:
                    score_left = score_one * (im - ia + 1)
                else:
                    score_left = scores[(ia, im)]
                if iz - im < 3:
                    score_right = score_one * (iz - im)
                else:
                    score_right = scores[(im + 1, iz)]

                if score_left + score_right < best_score:
                    best_score = score_left + score_right
                    cut_point = im
            scores[(ia, iz)] = best_score
            cut_points[(ia, iz)] = cut_point

    return scores, cut_points


def recover_splits_rec(cut_points, ia, iz):
    if (ia, iz) in cut_points:
        if cut_points[(ia, iz)] is None:
            return [(ia, iz)]
        else:
            im = cut_points[(ia, iz)]
            if im >= 0:
                return recover_splits_rec(cut_points, ia, im) + recover_splits_rec(
                    cut_points, im + 1, iz
                )
    return []


def compute_cycles_dyn(S_a, n_event_tot):
    scores, cut_points = get_table_dyn(S_a, n_event_tot)

    cycles = list()
    covered = set()
    for si, s in enumerate(cut_points):
        if s[1] - s[0] > 3:
            pass


# TODO : inherit MDLOptimizer
class PeriodicCycleMiner(BaseMiner):
    def fit(self, S):
        if not isinstance(S, pd.Series) or not isinstance(S.index, pd.DatetimeIndex):
            raise TypeError("S must be a Series with a datetime index")

        alpha_groups = S.groupby(S.values)
        alpha_sizes = alpha_groups.apply(len)

        # TODO