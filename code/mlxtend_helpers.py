"""
mlxtend_helpers.py
---------------------
Shared conversion helpers between this codebase's plain transaction-
list format and mlxtend's required one-hot-encoded DataFrame format,
used by both apriori.py and fpgrowth.py so both wrap mlxtend
consistently and return the same {frozenset: support_count} format
used throughout the rest of this codebase.
"""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def transactions_to_onehot(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def to_support_count_dict(result_df, n_transactions):
    return {
        frozenset(itemset): round(support * n_transactions)
        for itemset, support in zip(result_df["itemsets"], result_df["support"])
    }
