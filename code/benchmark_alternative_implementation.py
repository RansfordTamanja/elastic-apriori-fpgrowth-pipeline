"""
benchmark_alternative_implementation.py
------------------------------------------
Compares mlxtend's Apriori against a genuinely independent, separately-
maintained Apriori implementation (the `efficient-apriori` PyPI
package, unrelated to mlxtend's codebase), directly addressing the
reviewer's point: "the Apriori vs FP-Growth result concerns mlxtend,
not the algorithms themselves... benchmark at least one other
implementation."

This does NOT resolve whether FP-Growth would also be faster or slower
under a different implementation (efficient-apriori implements only
Apriori, not FP-Growth, so this is not a full second data point for
the paper's core comparison) -- but it directly tests a narrower,
still-useful question: is mlxtend's Apriori specifically fast or slow
relative to another real, independent Apriori implementation, which
bears on how much of the paper's finding is "Apriori is inherently
well-suited to vectorization" versus "mlxtend's specific Apriori
implementation happens to be well-optimized."

Setup: pip install efficient-apriori
Run: python benchmark_alternative_implementation.py
"""
import time
import numpy as np

from apriori import apriori as mlxtend_apriori
from data_sim import generate_transactions

try:
    from efficient_apriori import apriori as ea_apriori
except ImportError:
    raise ImportError("Run: pip install efficient-apriori")


def run_comparison(volumes, support, n_trials=3):
    rows = []
    for n in volumes:
        txns = generate_transactions(n, seed=0)
        txns_as_tuples = [tuple(t) for t in txns]  # efficient-apriori's expected format

        mlx_times = []
        for _ in range(n_trials):
            t0 = time.time()
            mlx_result = mlxtend_apriori(txns, min_support=support)
            mlx_times.append(time.time() - t0)

        ea_times = []
        for _ in range(n_trials):
            t0 = time.time()
            ea_itemsets, _ = ea_apriori(txns_as_tuples, min_support=support, min_confidence=1.0)
            ea_times.append(time.time() - t0)

        mlx_time = float(np.median(mlx_times))
        ea_time = float(np.median(ea_times))

        # Correctness cross-check: both should find the same itemsets,
        # modulo how each library represents/orders items
        mlx_itemsets = set(mlxtend_apriori(txns, min_support=support).keys())
        ea_flat = set()
        for length_level in ea_itemsets.values():
            for itemset in length_level:
                ea_flat.add(frozenset(itemset))
        itemsets_match = mlx_itemsets == ea_flat

        rows.append(dict(n_transactions=n, mlxtend_time=mlx_time,
                          efficient_apriori_time=ea_time,
                          mlxtend_vs_efficient_apriori_ratio=mlx_time / ea_time if ea_time > 0 else np.nan,
                          n_itemsets_mlxtend=len(mlx_itemsets), n_itemsets_efficient_apriori=len(ea_flat),
                          itemsets_match=itemsets_match))
        print(f"  n={n:7,}: mlxtend={mlx_time:.4f}s  efficient-apriori={ea_time:.4f}s  "
              f"ratio={mlx_time/ea_time if ea_time>0 else float('nan'):.2f}x  "
              f"itemsets match={itemsets_match} ({len(mlx_itemsets)} vs {len(ea_flat)})")
    return rows


if __name__ == "__main__":
    print("Comparing mlxtend's Apriori against efficient-apriori (an independent, "
          "separately-maintained implementation), same synthetic data, same support.\n")
    VOLUMES = [100, 1000, 5000, 10000]
    SUPPORT = 0.02
    rows = run_comparison(VOLUMES, SUPPORT)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("table_alternative_implementation_comparison.csv", index=False)
    print(f"\nSaved to table_alternative_implementation_comparison.csv")

    if not all(r["itemsets_match"] for r in rows):
        print("\nWARNING: itemset sets did not match at every volume tested -- "
              "investigate before drawing conclusions from the timing comparison, "
              "since a runtime comparison between implementations that disagree on "
              "output would not be meaningful (the same correctness precondition "
              "the paper already applies to its own mlxtend Apriori vs FP-Growth "
              "comparison).")
    else:
        print("\nItemsets matched between mlxtend and efficient-apriori at every "
              "volume tested -- both implementations agree on WHAT is frequent; "
              "the timing comparison above is therefore meaningful.")
