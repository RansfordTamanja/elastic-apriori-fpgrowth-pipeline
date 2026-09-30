"""
fpgrowth.py
------------
Wraps mlxtend's fpgrowth() implementation (Han, Pei & Yin, 2000), used
throughout this study as the comparison point against Apriori. Returns
{frozenset(itemset): support_count}, matching apriori.py's format
exactly, so results can be compared directly and no other file in this
codebase needs to change.

Apriori and FP-Growth are both deterministic algorithms with one
well-defined correct answer for a given transaction set and support
threshold, so the two functions here should produce identical
frequent itemsets to each other (verified directly below).
"""
from mlxtend.frequent_patterns import fpgrowth as _mlxtend_fpgrowth
from mlxtend_helpers import transactions_to_onehot, to_support_count_dict


def fpgrowth(transactions, min_support):
    df = transactions_to_onehot(transactions)
    result_df = _mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
    return to_support_count_dict(result_df, len(transactions))


if __name__ == "__main__":
    from data_sim import generate_transactions
    from apriori import apriori

    txns = generate_transactions(200, seed=0)
    result_fp = fpgrowth(txns, min_support=0.1)
    result_ap = apriori(txns, min_support=0.1)

    print(f"FP-Growth found {len(result_fp)} frequent itemsets")
    print(f"Apriori found {len(result_ap)} frequent itemsets")
    print("Results identical:", set(result_fp.keys()) == set(result_ap.keys()) and
          all(result_fp[k] == result_ap[k] for k in result_fp))
