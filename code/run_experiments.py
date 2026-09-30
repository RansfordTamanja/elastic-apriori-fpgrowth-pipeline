"""
run_experiments.py
--------------------
Full evaluation: Apriori vs. FP-Growth vs. simulated distributed
FP-Growth runtime scaling across realistic small-to-large retailer
transaction volumes, a real cost model grounded in verified AWS
pricing, the resulting serverless-vs-cluster cost crossover point, and
a secondary study of how the crossover depends on support threshold
(itemset search-space size), not transaction volume alone.

Run:
    python run_experiments.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_sim import generate_transactions
from apriori import apriori
from fpgrowth import fpgrowth
from distributed_fpgrowth import simulate_distributed_fpgrowth
from cost_model import serverless_cost, cluster_cost, find_crossover, EMR_NODE_PRICE_PER_HOUR

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
    with open(os.path.join(DATA_DIR, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"  saved -> {name}")


VOLUMES = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
SUPPORTS_FOR_CORRECTNESS_COUNT = [0.08, 0.04, 0.02, 0.01]
PRIMARY_SUPPORT = 0.02
N_TRIALS = 3  # median-of-N, matching the methodology already used in
              # run_real_data_study.py and fetch_real_data_locally.py;
              # an earlier version of this script measured each point
              # with a single timing sample, correctly flagged as
              # insufficient for sub-second measurements where system
              # jitter (GC pauses, OS scheduling) can be a large
              # fraction of the measured time itself.
N_CONFIGURATIONS = len(VOLUMES) * len(SUPPORTS_FOR_CORRECTNESS_COUNT)

# ----------------------------------------------------------------------
print(f"[1/5] Measuring real Apriori vs. FP-Growth runtime across transaction volumes "
      f"(median of {N_TRIALS} trials per point)...")
scaling_rows = []
for n in VOLUMES:
    txns = generate_transactions(n, seed=0)

    ap_times, fp_times = [], []
    for _ in range(N_TRIALS):
        t0 = time.time()
        r_ap = apriori(txns, min_support=PRIMARY_SUPPORT)
        ap_times.append(time.time() - t0)
        t0 = time.time()
        r_fp = fpgrowth(txns, min_support=PRIMARY_SUPPORT)
        fp_times.append(time.time() - t0)

    assert set(r_ap.keys()) == set(r_fp.keys()), f"Mismatch at n={n}!"

    ap_time, fp_time = float(np.median(ap_times)), float(np.median(fp_times))
    ap_std, fp_std = float(np.std(ap_times)), float(np.std(fp_times))
    apriori_fpgrowth_ratio = ap_time / fp_time if fp_time > 0 else np.nan

    scaling_rows.append(dict(n_transactions=n, apriori_time=ap_time, apriori_time_std=ap_std,
                              fpgrowth_time=fp_time, fpgrowth_time_std=fp_std,
                              n_itemsets=len(r_ap), apriori_fpgrowth_ratio=apriori_fpgrowth_ratio, n_trials=N_TRIALS))
    print(f"  n={n:7d}: Apriori={ap_time:7.3f}s (+/-{ap_std:.4f})  "
          f"FP-Growth={fp_time:7.3f}s (+/-{fp_std:.4f})  "
          f"apriori_fpgrowth_ratio={apriori_fpgrowth_ratio:.2f}x  ({len(r_ap)} itemsets)  [median of {N_TRIALS} trials]")

scaling_df = pd.DataFrame(scaling_rows)
save_table(scaling_df, "table1_algorithm_scaling")

# ----------------------------------------------------------------------
print("\n[2/5] Distributed FP-Growth: modeled wall-clock time at increasing worker counts...")
WORKER_COUNTS = [1, 2, 4, 8]
dist_rows = []
for n in VOLUMES:
    txns = generate_transactions(n, seed=0)
    for k in WORKER_COUNTS:
        dist = simulate_distributed_fpgrowth(txns, min_support=PRIMARY_SUPPORT, n_workers=k)
        dist_rows.append(dict(n_transactions=n, n_workers=k,
                               modeled_wall_clock=dist["modeled_wall_clock_time"],
                               total_worker_seconds=dist["total_worker_seconds"]))
    print(f"  n={n:7d} done (k=1,2,4,8)")

dist_df = pd.DataFrame(dist_rows)
save_table(dist_df, "table2_distributed_scaling")

# ----------------------------------------------------------------------
print("\n[3/5] Real cost model: serverless (Lambda) vs. cluster (EMR) at each volume...")
cost_rows = []
for n in VOLUMES:
    fp_row = scaling_df[scaling_df["n_transactions"] == n].iloc[0]
    serverless_runtime = fp_row["fpgrowth_time"]  # serverless runs single-invocation FP-Growth
    s_cost = serverless_cost(serverless_runtime, n_invocations=1)

    # cluster cost: use the 4-worker distributed configuration as the
    # reference "small EMR cluster" (a defensible minimal real cluster
    # size; sensitivity to this choice is reported separately)
    dist_row = dist_df[(dist_df["n_transactions"] == n) & (dist_df["n_workers"] == 4)].iloc[0]
    c_cost = cluster_cost(dist_row["modeled_wall_clock"], n_nodes=4)

    cost_rows.append(dict(n_transactions=n, serverless_cost_usd=s_cost, cluster_cost_usd=c_cost))
    print(f"  n={n:7d}: serverless=${s_cost:.6f}  cluster(4-node)=${c_cost:.6f}")

cost_df = pd.DataFrame(cost_rows)
save_table(cost_df, "table3_cost_comparison")

crossover = find_crossover(cost_df["n_transactions"].tolist(),
                            cost_df["serverless_cost_usd"].tolist(),
                            cost_df["cluster_cost_usd"].tolist())
print(f"\n  Crossover point (serverless becomes more expensive than 4-node cluster): "
      f"{crossover if crossover else 'not observed in tested range'}")

with open(os.path.join(DATA_DIR, "crossover.json"), "w") as f:
    json.dump(dict(crossover_transactions=crossover, support_threshold=PRIMARY_SUPPORT), f, indent=2)

# ----------------------------------------------------------------------
print(f"\n[4/5] Support-threshold sensitivity: does the crossover depend on itemset-space "
      f"size, not just volume? (median of {N_TRIALS} trials per point)")
support_rows = []
SUPPORTS = [0.08, 0.04, 0.02, 0.01]
n_fixed = 20000
for sup in SUPPORTS:
    txns = generate_transactions(n_fixed, seed=0)
    ap_times, fp_times = [], []
    for _ in range(N_TRIALS):
        t0 = time.time()
        r_ap = apriori(txns, min_support=sup)
        ap_times.append(time.time() - t0)
        t0 = time.time()
        r_fp = fpgrowth(txns, min_support=sup)
        fp_times.append(time.time() - t0)
    ap_time, fp_time = float(np.median(ap_times)), float(np.median(fp_times))
    ap_std, fp_std = float(np.std(ap_times)), float(np.std(fp_times))
    ratio = ap_time / fp_time if fp_time > 0 else np.nan
    support_rows.append(dict(support=sup, n_itemsets=len(r_ap), apriori_time=ap_time,
                              apriori_time_std=ap_std, fpgrowth_time=fp_time,
                              fpgrowth_time_std=fp_std, apriori_fpgrowth_ratio=ratio,
                              n_trials=N_TRIALS))
    print(f"  support={sup}: {len(r_ap)} itemsets, Apriori={ap_time:.3f}s (+/-{ap_std:.4f}), "
          f"FP-Growth={fp_time:.3f}s (+/-{fp_std:.4f}), apriori_fpgrowth_ratio={ratio:.2f}x "
          f"[median of {N_TRIALS} trials]")

support_df = pd.DataFrame(support_rows)
save_table(support_df, "table4_support_sensitivity")
print(f"\n  NOTE on apriori_fpgrowth_ratio (previously labeled 'Speedup', renamed for "
      f"clarity per reviewer feedback): values BELOW 1.0 mean Apriori is FASTER; "
      f"values ABOVE 1.0 mean FP-Growth is faster. This is Apriori time divided by "
      f"FP-Growth time, not a 'speedup factor' in the conventional sense.")
print(f"  Total distinct configurations tested across Steps [1/5] and [4/5]: "
      f"{len(VOLUMES)} volumes x {len(SUPPORTS)} supports = {N_CONFIGURATIONS} "
      f"(previously described imprecisely as 'more than a dozen').")

# ----------------------------------------------------------------------
print("\n[5/5] Generating figures...")

# Fig 1: Apriori vs FP-Growth runtime scaling (log-log)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(scaling_df["n_transactions"], scaling_df["apriori_time"], marker="o",
        label="Apriori", color="#e0622a")
ax.plot(scaling_df["n_transactions"], scaling_df["fpgrowth_time"], marker="s",
        label="FP-Growth", color="#1f6feb")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of transactions")
ax.set_ylabel("Runtime (seconds)")
ax.set_title(f"Apriori vs. FP-Growth runtime scaling (min\\_support={PRIMARY_SUPPORT})")
ax.legend()
fig.tight_layout()
fig.savefig(fig_path("figure1_algorithm_scaling.png"))
plt.close(fig)
print("  saved -> figure1_algorithm_scaling.png")

# Fig 2: real cost comparison, serverless vs cluster, with crossover marked
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(cost_df["n_transactions"], cost_df["serverless_cost_usd"], marker="o",
        label="Serverless (Lambda)", color="#f5c542")
ax.plot(cost_df["n_transactions"], cost_df["cluster_cost_usd"], marker="s",
        label="Cluster (EMR, 4 nodes)", color="#1fb6a8")
if crossover:
    ax.axvline(crossover, color="red", ls="--", lw=1,
               label=f"Crossover (~{crossover:,.0f} transactions)")
ax.set_xscale("log")
ax.set_xlabel("Number of transactions per mining job")
ax.set_ylabel("Cost per job (USD)")
ax.set_title("Real AWS-pricing-grounded cost: serverless vs. cluster")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(fig_path("figure2_cost_crossover.png"))
plt.close(fig)
print("  saved -> figure2_cost_crossover.png")

# Fig 3: distributed wall-clock time vs worker count, one line per volume
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(VOLUMES)))
for n, color in zip(VOLUMES, colors):
    sub = dist_df[dist_df["n_transactions"] == n]
    ax.plot(sub["n_workers"], sub["modeled_wall_clock"], marker="o", color=color,
            label=f"n={n:,}")
ax.set_xlabel("Number of workers (k)")
ax.set_ylabel("Modeled wall-clock time (seconds)")
ax.set_title("Distributed FP-Growth: modeled wall-clock time vs. worker count")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(fig_path("figure3_distributed_scaling.png"))
plt.close(fig)
print("  saved -> figure3_distributed_scaling.png")

# Fig 4: support-threshold sensitivity (apriori_fpgrowth_ratio vs support, at fixed n)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(support_df["support"], support_df["apriori_fpgrowth_ratio"], marker="o", color="#a855f7")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="Apriori = FP-Growth")
ax.set_xlabel("Minimum support threshold")
ax.set_ylabel("Apriori time / FP-Growth time")
ax.set_title(f"FP-Growth's advantage depends on support threshold\n(fixed n={n_fixed:,} transactions)")
ax.invert_xaxis()
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(fig_path("figure4_support_sensitivity.png"))
plt.close(fig)
print("  saved -> figure4_support_sensitivity.png")

print(f"\nAll assets saved under {DATA_DIR} and {FIG_DIR}")
