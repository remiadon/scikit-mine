"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
"""

# Author: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from roaringbitmap import RoaringBitmap
from sortedcontainers import SortedDict

from ..base import BaseMiner


def _check_min_supp(min_supp):
    if isinstance(min_supp, int):
        if min_supp < 1:
            raise ValueError('Minimum support must be strictly positive')
    elif isinstance(min_supp, float):
        if min_supp < 0 or min_supp > 1:
            raise ValueError('Minimum support must be between 0 and 1')
    else:
        raise TypeError('Mimimum support must be of type int or float')
    return min_supp


class LCM(BaseMiner):
    """
    Linear time Closed item set Miner.

    Parameters
    ----------

    min_supp: int or float, default=2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support

    n_jobs : int, default=1
        The number of jobs to use for the computation. Parallelism is done by launching a job
        to compute a descent on each root node of the closed itemset lattice.
        Processes are preffered over threads.

    Examples
    --------

    >>> from skmine.itemsets import LCM
    >>> from skmine.datasets.fimi import fetch_chess
    >>> chess = fetch_chess()
    >>> lcm = LCM(min_supp=2000)
    >>> patterns = lcm.fit_transform(chess)
    >>> patterns.head()
        itemset support
    0      (58)    3195
    1  (11, 58)    2128
    2  (15, 58)    2025
    3  (17, 58)    2499
    4  (21, 58)    2224
    >>> patterns[patterns.itemset.map(len) > 3]  # only keeps itemsets longer than 3
    """
    def __init__(self, *, min_supp=2, n_jobs=1):
        _check_min_supp(min_supp)
        self.min_supp = min_supp  # provided by user
        self._min_supp = _check_min_supp(self.min_supp)
        self.item_to_tids = SortedDict()
        self.n_transactions = 0
        self.ctr = 0
        self.n_jobs = n_jobs

    def fit(self, D):
        """fit LCM on the transactional database
        This simply iterates over transactions of D in order to keep
        track of every item and transactions ids related

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        Returns
        -------
        self: LCM
            a reference to the model itself

        """
        for transaction in D:
            for item in transaction:
                if item in self.item_to_tids:
                    self.item_to_tids[item].add(self.n_transactions)
                else:
                    self.item_to_tids[item] = RoaringBitmap([self.n_transactions])
            self.n_transactions += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions

        return self

    partial_fit = fit

    def get_binary_matrix(self):
        """Construct a binary matrix from previously seen transactions
        """
        valid_items = sorted((k for k, v in self.item_to_tids.items() if len(v) >= self._min_supp))
        shape = (self.n_transactions, len(valid_items))
        m = np.zeros(shape, dtype=np.bool)
        # semi-vectorized initialisation
        # in practice len(self._items) is not big so that's OK
        for col_idx, item in enumerate(valid_items):
            tids = self.item_to_tids[item]
            m[tids, col_idx] = True
        return pd.DataFrame(m, columns=valid_items)

    def fit_transform(self, D):
        """fit LCM on the transactional database, and return the set of
        closed itemsets in this database, with respect to the minium support

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        Returns
        -------
        pd.DataFrame:
            DataFrame with the following columns
                ==========  =================================
                itemset     a `frozenset` of co-occured items
                support     frequence for this itemset
                ==========  =================================

        """
        self.fit(D)

        m = self.get_binary_matrix()
        empty_df = pd.DataFrame(columns=['itemset', 'support'])

        # reverse order of support
        sorted_items = sorted(m.columns, key=lambda e: len(self.item_to_tids[e]), reverse=True)
        dfs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(self._explore_item)(item, m) for item in sorted_items
        )

        dfs.append(empty_df) # make sure we have something to concat
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _explore_item(self, item, matrix):
        tids = self.item_to_tids[item]
        it = self._inner(frozenset([item]), tids, item, matrix)
        df = pd.DataFrame(data=it, columns=['itemset', 'support'])
        df.support = df.support.astype(np.uint32)
        if not df.empty:
            print('LCM found {} new itemsets from item : {}'.format(len(df), item))
        return df

    def _inner(self, p, tids, limit, DB):
        # project and reduce DB w.r.t P and limit
        CDB = DB.iloc[tids]
        freqs = CDB.sum()

        split_crit = (freqs == len(tids))
        cp = freqs[split_crit].index
        if cp.max() == limit:
            cp = frozenset(cp)
            p_prime = p | cp
            yield p_prime, len(tids)

            cands = freqs[~split_crit]
            cands = cands.loc[:limit]
            cands = cands[cands >= self._min_supp]

            new_DB = None
            for cand in cands.index:
                if new_DB is None:
                    new_DB = DB.drop(cp, axis=1, errors='ignore')
                new_tids = tids.intersection(self.item_to_tids[cand])
                yield from self._inner(p_prime, new_tids, cand, new_DB)
