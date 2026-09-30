"""
run_all.py

Runs the full elastic pipeline study: environment capture, the
synthetic benchmark, the mechanism profiling, the alternative-
implementation comparison, and (if you provide the required data
files) the real-data validations.

Usage:
    pip install -r requirements.txt
    python run_all.py

Some steps require data files this repository does not include (see
README.md): the Kaggle Market Basket Optimisation CSV for
run_real_data_study.py, and network access to UCI's servers for
fetch_real_data_locally.py. Both are skipped automatically with a
clear message if their prerequisites aren't met, rather than crashing
the whole run.
"""
import subprocess
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(THIS_DIR, "code")


def run_step(description, script, required_file=None, required_file_desc=None):
    print("=" * 70)
    print(description)
    print("=" * 70)
    if required_file and not os.path.exists(os.path.join(CODE_DIR, required_file)):
        print(f"SKIPPED: {required_file_desc}")
        print()
        return
    subprocess.run([sys.executable, script], check=True, cwd=CODE_DIR)
    print()


run_step("STEP 1/6: Capturing hardware/software environment info", "env_info.py")

run_step("STEP 2/6: Main synthetic benchmark (Tables 1-4, Figures 1-4): "
         "Apriori vs. FP-Growth scaling, distributed simulation, cost model, "
         "support sensitivity -- all with median-of-3 trials and variance",
         "run_experiments.py")

run_step("STEP 3/6: Profiling Apriori and FP-Growth internals directly "
         "(addresses reviewer point 4: profile, don't only infer)",
         "profile_mechanism.py")

run_step("STEP 4/6: Comparing mlxtend's Apriori against an independent "
         "implementation (efficient-apriori) (addresses reviewer point 3)",
         "benchmark_alternative_implementation.py")

run_step("STEP 5/6: Real-data validation on the Kaggle Market Basket dataset "
         "(requires real_data/Market_Basket_Optimisation.csv -- see README)",
         "run_real_data_study.py",
         required_file="real_data/Market_Basket_Optimisation.csv",
         required_file_desc="real_data/Market_Basket_Optimisation.csv not found. "
                             "Download it from Kaggle (see README.md) and place it "
                             "at code/real_data/Market_Basket_Optimisation.csv, "
                             "then re-run this step directly: "
                             "cd code && python run_real_data_study.py")

print("=" * 70)
print("STEP 6/6: Full-scale real-data extension on the UCI Online Retail "
      "dataset (requires network access to UCI's servers, or a local cache "
      "file from a previous successful run)")
print("=" * 70)
try:
    subprocess.run([sys.executable, "fetch_real_data_locally.py"], check=True, cwd=CODE_DIR)
except subprocess.CalledProcessError:
    print("Step 6 failed -- this commonly means no network access to UCI's "
          "servers from this machine. Run this step directly on a machine "
          "with internet access: cd code && python fetch_real_data_locally.py")

print("\nDone. See ./data for CSVs/JSON and ./figures for PNGs.")
