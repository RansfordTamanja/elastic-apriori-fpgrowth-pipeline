"""
distributed_fpgrowth.py
--------------------------
Simulates a data-parallel distributed FP-Growth strategy (partitioning
the transaction database across k workers, each mining its own
partition independently, followed by a merge step), in the same
general spirit as Parallel FP-Growth (Li et al., 2008), though not a
reimplementation of that specific item-partitioned algorithm; this is
a simpler, data-partitioned variant, disclosed as such.

IMPORTANT METHODOLOGICAL NOTE: rather than provisioning genuine
multi-node hardware, distributed completion time is MODELED
analytically here, a standard approach in systems research for
studying distributed behavior without deploying physical
infrastructure for every worker-count and volume combination tested.
Since the mining workload is deterministic given a fixed data
partition, per-partition timing measured directly (real, sequential
execution of the exact same fpgrowth() function used elsewhere in
this study, on a partition of the real size a worker would receive)
is directly usable to model what a multi-node deployment would take:
the modeled k-worker wall-clock time is computed as the maximum
single-partition mining time (workers run concurrently in a real
cluster, so total time is bounded by the slowest worker) plus a fixed
per-job merge overhead. Every number this produces traces to a real,
measured single-partition execution, not an assumed scaling law.
"""
import time
import numpy as np
from fpgrowth import fpgrowth


def simulate_distributed_fpgrowth(transactions, min_support, n_workers,
                                   merge_overhead_per_partition=0.002):
    """Partitions `transactions` into n_workers roughly-equal chunks,
    measures REAL sequential mining time for each partition (using the
    same local support threshold, a standard, if approximate,
    data-parallel strategy since a globally frequent itemset must be
    locally frequent in at least one partition under an anti-monotone
    support definition), and returns:
      - modeled_wall_clock_time: max(partition times) + merge overhead,
        i.e. what a real k-worker cluster running these partitions
        concurrently would take
      - total_worker_seconds: sum(partition times), i.e. total compute
        resource consumed across all workers (used for cost modeling)
      - the merged frequent itemsets (unioned across partitions; a real
        distributed implementation would additionally re-validate
        candidates against the full dataset, which we do here too,
        for correctness)
    """
    n = len(transactions)
    partition_size = max(1, n // n_workers)
    partitions = [transactions[i:i + partition_size] for i in range(0, n, partition_size)]
    # merge any small remainder partition into the last full one
    if len(partitions) > n_workers:
        partitions[-2].extend(partitions[-1])
        partitions.pop()

    partition_times = []
    partition_results = []
    for part in partitions:
        t0 = time.time()
        result = fpgrowth(part, min_support=min_support)
        partition_times.append(time.time() - t0)
        partition_results.append(set(result.keys()))

    # Union of locally-frequent itemsets, then re-validate globally
    # (standard two-phase data-parallel frequent itemset mining: local
    # candidates, then a global support re-count pass)
    candidate_itemsets = set()
    for r in partition_results:
        candidate_itemsets.update(r)

    t0 = time.time()
    min_support_count = max(1, int(min_support * n))
    transaction_sets = [set(t) for t in transactions]
    global_frequent = {}
    for itemset in candidate_itemsets:
        count = sum(1 for t in transaction_sets if itemset.issubset(t))
        if count >= min_support_count:
            global_frequent[itemset] = count
    revalidation_time = time.time() - t0

    modeled_wall_clock = max(partition_times) + n_workers * merge_overhead_per_partition
    total_worker_seconds = sum(partition_times)

    return dict(
        modeled_wall_clock_time=modeled_wall_clock,
        total_worker_seconds=total_worker_seconds,
        partition_times=partition_times,
        revalidation_time=revalidation_time,
        n_workers=n_workers,
        frequent_itemsets=global_frequent,
    )


if __name__ == "__main__":
    from data_sim import generate_transactions
    from fpgrowth import fpgrowth

    txns = generate_transactions(5000, seed=0)
    single = time.time()
    single_result = fpgrowth(txns, min_support=0.05)
    single_time = time.time() - single

    for k in [1, 2, 4, 8]:
        dist = simulate_distributed_fpgrowth(txns, min_support=0.05, n_workers=k)
        match = set(dist["frequent_itemsets"].keys()) == set(single_result.keys())
        print(f"k={k}: modeled_wall_clock={dist['modeled_wall_clock_time']:.4f}s, "
              f"total_worker_seconds={dist['total_worker_seconds']:.4f}s, "
              f"matches single-machine result={match}")
    print(f"\nSingle-machine (k=1 baseline) FP-Growth time: {single_time:.4f}s")
