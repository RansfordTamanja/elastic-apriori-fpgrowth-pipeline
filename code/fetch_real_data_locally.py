"""
fetch_real_data_locally.py
------------------------------
Run this on a machine with internet access. Downloads the real, full
UCI Online Retail dataset (541,909 rows, Chen 2015, CC BY 4.0) via the
official ucimlrepo package, groups it into real transaction baskets by
InvoiceNo, and runs the same apriori()/fpgrowth()/cost_model pipeline
used throughout this paper against it at real transaction volumes far
beyond the 7,501-transaction Kaggle dataset used in the paper's main
real-data validation section.

This produces the same tables and figures as run_real_data_study.py,
saved under ../data/ and ../figures/, so results can be directly
compared and merged into the paper's real-data section.

Setup: pip install ucimlrepo pandas numpy matplotlib
Run: python fetch_real_data_locally.py
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from apriori import apriori
from fpgrowth import fpgrowth
from cost_model import serverless_cost, cluster_cost

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                      "axes.spines.top": False, "axes.spines.right": False})


def fig_path(name):
    return os.path.join(FIG_DIR, name)


def save_table(df, name):
    df.to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
    try:
        with open(os.path.join(DATA_DIR, f"{name}.md"), "w") as f:
            f.write(df.to_markdown(index=False))
    except ImportError:
        pass  # tabulate not installed; CSV is still saved
    print(f"  saved -> {name}")


# ----------------------------------------------------------------------
print("[1/4] Fetching the real, full UCI Online Retail dataset...")
CACHE_FILE = "uci_online_retail_transactions_cache.pkl"
if os.path.exists(CACHE_FILE):
    print(f"  Found local cache ({CACHE_FILE}), loading instead of re-downloading...")
    transactions_all = pd.read_pickle(CACHE_FILE)
    print(f"  Loaded {len(transactions_all):,} real transaction baskets from cache.")
else:
    online_retail = fetch_ucirepo(id=352)
    # InvoiceNo and StockCode are classified by ucimlrepo's own metadata as
    # "ID" columns, not "Feature" columns, so they live in .data.ids, not
    # .data.features; confirmed directly via diagnose_columns.py rather
    # than assumed. We combine both into one DataFrame since this paper's
    # pipeline needs InvoiceNo (to group into transaction baskets) and
    # Description (the actual product name) together.
    df = pd.concat([online_retail.data.ids, online_retail.data.features], axis=1)
    print(f"  Fetched {len(df):,} real rows from the UCI Online Retail dataset.")

    # Standard cleaning: drop cancellations (InvoiceNo starting with 'C')
    # and non-product administrative line items, matching the paper's
    # documented cleaning methodology exactly.
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "DOT", "CRUK", "AMAZONFEE"}
    df = df[~df["StockCode"].astype(str).isin(NON_PRODUCT_CODES)]
    df = df.dropna(subset=["Description"])

    transactions_all = df.groupby("InvoiceNo")["Description"].apply(
        lambda x: sorted(set(x.str.strip()))
    ).tolist()
    print(f"  Grouped into {len(transactions_all):,} real transaction baskets "
          f"after removing cancellations and non-product line items.")

    pd.to_pickle(transactions_all, CACHE_FILE)
    print(f"  Cached to {CACHE_FILE} so future runs skip the download step.")

# ----------------------------------------------------------------------
print("\n[2/4] Volume scaling at full real-data scale (random-shuffled subsets, median of 3 trials)...")
# Random-shuffled subsets, not sequential prefixes, matching the
# paper's Kaggle real-data methodology exactly, specifically to rule
# out any effect of transaction ordering in the raw log. Median of 3
# timing trials per point, matching the paper's jitter-reduction
# methodology for the smaller Kaggle dataset.
N_MAX = len(transactions_all)
REAL_VOLUMES = sorted(set([100, 500, 1000, 5000, 10000, 25000, 50000,
                            100000, 250000, N_MAX]))
REAL_VOLUMES = [v for v in REAL_VOLUMES if v <= N_MAX]
SUPPORT = 0.02
rng = np.random.default_rng(0)

scaling_rows = []
for n in REAL_VOLUMES:
    if n == N_MAX:
        # BUG FIX: previously this always went through rng.choice(),
        # even at n == N_MAX, which technically reshuffles the entire
        # dataset into a NEW list object rather than using
        # transactions_all directly. Since apriori/fpgrowth are
        # order-invariant for CORRECTNESS (confirmed: itemset counts
        # matched exactly before and after this fix), that reshuffle
        # didn't change WHAT was found, only introduced a spurious,
        # unnecessary difference in WALL-CLOCK TIMING relative to
        # Step 3 below, which uses transactions_all directly without
        # any reshuffling. This is exactly why Table VI and Table VII
        # previously reported different runtimes (5.829s vs. 3.795s)
        # for what is supposed to be the identical configuration
        # (n=N_MAX, support=0.02). Using transactions_all directly
        # here, with no resampling, makes both tables measure the
        # same thing the same way at their one overlapping point.
        sample = transactions_all
    else:
        idx = rng.choice(len(transactions_all), size=n, replace=False)
        sample = [transactions_all[i] for i in idx]

    ap_times, fp_times = [], []
    for trial in range(3):
        t0 = time.time()
        r_ap = apriori(sample, min_support=SUPPORT)
        ap_times.append(time.time() - t0)
        t0 = time.time()
        r_fp = fpgrowth(sample, min_support=SUPPORT)
        fp_times.append(time.time() - t0)
    ap_time, fp_time = float(np.median(ap_times)), float(np.median(fp_times))
    match = set(r_ap.keys()) == set(r_fp.keys())
    speedup = ap_time / fp_time if fp_time > 0 else float("nan")

    scaling_rows.append(dict(n_transactions=n, apriori_time=ap_time, fpgrowth_time=fp_time,
                              n_itemsets=len(r_ap), speedup=speedup, outputs_match=match))
    print(f"  n={n:>7,}: Apriori={ap_time:7.3f}s  FP-Growth={fp_time:7.3f}s  "
          f"speedup={speedup:.2f}x  ({len(r_ap)} itemsets)  "
          f"{'[MATCH]' if match else '[MISMATCH -- INVESTIGATE]'}")

scaling_df = pd.DataFrame(scaling_rows)
save_table(scaling_df, "table_real_fullscale_volume_scaling")

# ----------------------------------------------------------------------
print(f"\n[3/4] Support-threshold sensitivity at full real-data scale (n={N_MAX:,})...")
SUPPORTS = [0.08, 0.04, 0.02, 0.01]
support_rows = []
for sup in SUPPORTS:
    try:
        ap_times, fp_times = [], []
        for trial in range(3):
            t0 = time.time()
            r_ap = apriori(transactions_all, min_support=sup)
            ap_times.append(time.time() - t0)
            t0 = time.time()
            r_fp = fpgrowth(transactions_all, min_support=sup)
            fp_times.append(time.time() - t0)
        ap_time, fp_time = float(np.median(ap_times)), float(np.median(fp_times))
        match = set(r_ap.keys()) == set(r_fp.keys())
        speedup = ap_time / fp_time if fp_time > 0 else float("nan")
        support_rows.append(dict(support=sup, n_itemsets=len(r_ap), apriori_time=ap_time,
                                  fpgrowth_time=fp_time, speedup=speedup, outputs_match=match,
                                  apriori_oom=False))
        print(f"  support={sup}: {len(r_ap)} itemsets, Apriori={ap_time:.3f}s, "
              f"FP-Growth={fp_time:.3f}s, speedup={speedup:.2f}x "
              f"{'[MATCH]' if match else '[MISMATCH -- INVESTIGATE]'}")
    except MemoryError as e:
        # Genuine, informative finding, not a bug to hide: vectorized
        # Apriori (mlxtend's current default) must materialize a full
        # (n_transactions x n_candidate_itemsets) boolean matrix at
        # once; at low support and large real scale, this candidate
        # matrix can exceed available memory even though FP-Growth's
        # tree-based approach handles the identical data and support
        # threshold without issue. We record this directly as a real
        # result (a memory/time tradeoff introduced by the same 2019
        # vectorization that makes Apriori faster), not a crash to
        # suppress, and separately measure FP-Growth alone so the
        # comparison at this support level is not simply missing.
        print(f"  support={sup}: Apriori raised MemoryError (vectorized candidate matrix "
              f"too large at this scale/support) -- recording as a genuine finding")
        t0 = time.time()
        r_fp = fpgrowth(transactions_all, min_support=sup)
        fp_time = time.time() - t0
        print(f"    FP-Growth alone: {fp_time:.3f}s, {len(r_fp)} itemsets (completed with no issue)")
        support_rows.append(dict(support=sup, n_itemsets=len(r_fp), apriori_time=float("nan"),
                                  fpgrowth_time=fp_time, speedup=float("nan"), outputs_match=None,
                                  apriori_oom=True))

support_df = pd.DataFrame(support_rows)
save_table(support_df, "table_real_fullscale_support_sensitivity")

# Real cost comparison at full scale, hardest tested support
hardest = support_df.iloc[support_df["support"].idxmin()]
s_cost = serverless_cost(hardest["fpgrowth_time"], n_invocations=1)
c_cost = cluster_cost(hardest["fpgrowth_time"], n_nodes=4)
print(f"\n  Real cost at full scale (n={N_MAX:,}, support={hardest['support']}): "
      f"serverless=${s_cost:.6f}  cluster(4-node)=${c_cost:.6f}  "
      f"ratio={c_cost/s_cost:.1f}x")

# ----------------------------------------------------------------------
print("\n[4/4] Generating figures...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].plot(scaling_df["n_transactions"], scaling_df["speedup"], marker="o", color="#1f6feb")
axes[0].axhline(1.0, color="gray", ls=":", lw=1, label="Apriori = FP-Growth")
axes[0].set_xscale("log")
axes[0].set_xlabel("Number of real transactions")
axes[0].set_ylabel("Apriori time / FP-Growth time")
axes[0].set_title(f"Full-scale real data (up to {N_MAX:,} transactions)")
axes[0].legend(fontsize=8)

axes[1].plot(scaling_df["n_transactions"], scaling_df["n_itemsets"], marker="o", color="#059669")
axes[1].set_xscale("log")
axes[1].set_xlabel("Number of real transactions")
axes[1].set_ylabel("Number of frequent itemsets found")
axes[1].set_title(f"Itemset count vs. volume (min_support={SUPPORT})")

fig.tight_layout()
fig.savefig(fig_path("figure9_real_fullscale_scaling.png"))
plt.close(fig)
print("  saved -> figure9_real_fullscale_scaling.png")

print(f"\nAll full-scale real-data assets saved under {DATA_DIR} and {FIG_DIR}")
print(f"Total real transactions tested at full scale: {N_MAX:,}")
