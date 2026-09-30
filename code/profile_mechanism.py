"""
profile_mechanism.py
-----------------------
Profiles mlxtend's apriori() and fpgrowth() internals directly via
cProfile, rather than relying only on the changelog-based mechanistic
account and the low_memory=True diagnostic already in the paper. This
directly addresses the reviewer's request: "Profile both functions or
soften the abstract's 'tracing this to' claim."

This decomposes wall-clock time into per-function contributions,
letting you see directly whether time is actually concentrated in the
vectorized support-counting step (the paper's claimed mechanism) or
elsewhere (input validation, DataFrame construction, etc.), rather
than inferring this only from the low_memory=True test's aggregate
timing.

Run:
    python profile_mechanism.py
"""
import cProfile
import pstats
import io

from apriori import apriori
from fpgrowth import fpgrowth
from data_sim import generate_transactions


def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)  # top 15 functions by cumulative time
    return result, stream.getvalue()


def summarize_top_contributors(profile_text, label):
    """Extracts and prints just the function-name / cumulative-time
    columns from cProfile's verbose output, for quick comparison."""
    lines = profile_text.split("\n")
    print(f"\n--- Top time contributors: {label} ---")
    header_seen = False
    for line in lines:
        if "ncalls" in line and "cumtime" in line:
            header_seen = True
            continue
        if header_seen and line.strip():
            print(f"  {line.strip()}")


if __name__ == "__main__":
    N = 20000
    SUPPORT = 0.02
    print(f"Profiling Apriori and FP-Growth at n={N:,} transactions, support={SUPPORT}...")
    print("(This produces a detailed per-function breakdown, not just aggregate")
    print(" wall-clock time, to directly test WHERE time is spent in each algorithm.)\n")

    txns = generate_transactions(N, seed=0)

    _, ap_profile = profile_function(apriori, txns, min_support=SUPPORT)
    summarize_top_contributors(ap_profile, "Apriori")

    _, fp_profile = profile_function(fpgrowth, txns, min_support=SUPPORT)
    summarize_top_contributors(fp_profile, "FP-Growth")

    with open("profile_apriori_full.txt", "w") as f:
        f.write(ap_profile)
    with open("profile_fpgrowth_full.txt", "w") as f:
        f.write(fp_profile)

    print("\nFull profiles saved to profile_apriori_full.txt and "
          "profile_fpgrowth_full.txt. Look specifically for whether "
          "mlxtend's internal support-counting function (the specific "
          "step the paper's mechanistic account attributes the "
          "2019 vectorization to) actually dominates Apriori's "
          "profile, versus whether input validation, TransactionEncoder "
          "conversion, or another step contributes comparably or more.")
    print("\nThis is complementary to, not a replacement for, the paper's existing "
          "low_memory=True diagnostic (Section V-C): that test isolates the effect "
          "of DISABLING the vectorization; this profiling shows where time is spent "
          "WITH it enabled, the two together giving a more complete mechanistic "
          "picture than either alone.")
