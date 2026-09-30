"""
apriori.py
-----------
Wraps mlxtend's apriori() implementation (Agrawal & Srikant, 1994),
used as the baseline frequent-itemset-mining algorithm throughout
this study. Returns {frozenset(itemset): support_count}, the same
format used throughout the rest of this codebase, so no other file
needs to change.
"""
from mlxtend.frequent_patterns import apriori as _mlxtend_apriori
from mlxtend_helpers import transactions_to_onehot, to_support_count_dict


def apriori(transactions, min_support):
    df = transactions_to_onehot(transactions)
    result_df = _mlxtend_apriori(df, min_support=min_support, use_colnames=True)
    return to_support_count_dict(result_df, len(transactions))


if __name__ == "__main__":
    from data_sim import generate_transactions

    txns = generate_transactions(200, seed=0)
    result = apriori(txns, min_support=0.1)
    print(f"Found {len(result)} frequent itemsets at min_support=0.1")
    top5 = sorted(result.items(), key=lambda x: -x[1])[:5]
    for itemset, count in top5:
        print(f"  {set(itemset)}: support={count/len(txns):.3f}")
