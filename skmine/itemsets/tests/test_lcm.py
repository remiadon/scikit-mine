import pandas as pd
import pytest

from skmine.itemsets import LCM
from functools import partial

def next_(*args):
    return next(*args, None) or (None, None)

D = pd.Series([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 5],
    [2, 5],
    [1, 2, 4, 5, 6],
    [2, 4],
    [1, 4, 6],
    [3, 4, 6],
])

true_item_to_tids = {
    1 : {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3 : {0, 1, 6},
    4 : {0, 3, 4, 5, 6},
    5 : {0, 1, 2, 3},
    6 : {0, 3, 5, 6},
}

matrix = LCM(min_supp=3).fit(D).get_binary_matrix()

def test_binary_matrix():
    lcm = LCM(min_supp=3)
    lcm.fit(D)
    m = lcm.get_binary_matrix()
    assert m.columns.tolist() == list(range(1, 7))
    assert m.sum().sum() == 24


def test_lcm_fit():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]


def test_first_parent_limit_1():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([4, 6])
    tids = lcm.get_tids(p)

    ## pattern = {4, 6} -> first parent fails
    itemset, supp = next_(lcm._inner(p, tids, 1, matrix))
    assert itemset == None
    assert supp == None

    p = frozenset([1])
    tids = lcm.get_tids(p)
    # pattern = {1} -> first parent fails
    itemset, supp = next_(lcm._inner(p, tids, 1, matrix))
    assert itemset == None
    assert supp == None


def test_first_parent_limit_2():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([2])
    tids = lcm.get_tids(p)
    # pattern = {} -> first parent OK
    tids = lcm.item_to_tids[2]
    itemset, supp = next(lcm._inner(p, tids, 2, matrix), (None, None))
    assert itemset == frozenset([2])
    assert supp == 5


def test_first_parent_limit_3():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([3])
    tids = lcm.get_tids(p)
    itemset, supp = next(lcm._inner(p, tids, 3, matrix), (None, None))
    assert itemset == frozenset([3])
    assert supp == 3


def test_first_parent_limit_4():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([4])
    tids = lcm.get_tids(p)
    itemset, supp = next(lcm._inner(p, tids, 4, matrix), (None, None))
    assert itemset == frozenset([4])
    assert supp == 5


def test_first_parent_limit_5():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([5])
    tids = lcm.get_tids(p)
    itemset, supp = next(lcm._inner(p, tids, 5, matrix), (None, None))
    assert itemset == frozenset([2, 5])
    assert supp == 4


def test_first_parent_limit_6():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    p = frozenset([6])
    tids = lcm.get_tids(p)
    itemset, supp = next(lcm._inner(p, tids, 6, matrix), (None, None))
    assert itemset == frozenset([4, 6])
    assert supp == 4

    p = frozenset([1])
    tids = lcm.get_tids(p)
    itemset, supp = next(lcm._inner(p, tids, 6, matrix), (None, None))
    assert itemset == frozenset([1, 4, 6])
    assert supp == 3


def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_transform(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

    # 2. test with empty data
    lcm = LCM(min_supp=3)
    res = lcm.fit_transform([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty



def test_lcm_transform():
    lcm = LCM(min_supp=3)
    X = lcm.fit_transform(D)  # get new pattern set

    true_X = pd.DataFrame([
            [{2}, 5],
            [{4}, 5],
            [{2, 4}, 3],
            [{2, 5}, 4],
            [{4, 6}, 4],
            [{1, 4, 6}, 3],
            [{3}, 3],
    ], columns=['itemset', 'support'])

    true_X.loc[:, 'itemset'] = true_X.itemset.map(frozenset)

    for itemset, true_itemset in zip(X.itemset, true_X.itemset):
        assert itemset == true_itemset
    pd.testing.assert_series_equal(X.support, true_X.support, check_dtype=False)


def test_relative_support_errors():
    wrong_values = [-1, -100, 2.33, 150.55]
    for wrong_supp in wrong_values:
        with pytest.raises(ValueError):
            LCM(min_supp=wrong_supp)

    with pytest.raises(TypeError):
        LCM(min_supp='string minimum support')


def test_relative_support():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]

    assert round(lcm._min_supp) == 3
