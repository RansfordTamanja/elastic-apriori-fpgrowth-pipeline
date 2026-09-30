# An Elastic Cloud Pipeline for Small-to-Big Data Transition in Frequent Pattern Mining

Code for the paper of the same name, submitted to IEEE ICAST 2026 (Paper ID 39).

## What this update addresses

A detailed peer review raised 12 points. This update fixes everything
fixable at the code level; several points (AWS deployment framing,
missing citations, prose editing, the code repository link itself)
require changes to the paper text or your own input, not code, and are
listed at the bottom of this README as still open.

## Real bugs found and fixed in this update

1. **Table VI vs. Table VII reported different runtimes for what was
   supposed to be the identical configuration** (n=20,437 transactions,
   support=0.02): 5.829s vs. 3.795s. Root cause, found by reading the
   actual code: the volume-scaling step (`fetch_real_data_locally.py`)
   always resampled via `rng.choice()`, even when the sample size
   equaled the full dataset, creating a different Python list object
   than the support-sensitivity step's direct use of the cached
   transaction list. Itemset count matched exactly before and after
   this fix (361 in both cases, confirming mining is correctly
   order-invariant) -- only the measured wall-clock time differed, for
   a reason that had nothing to do with the actual algorithms. Fixed:
   the code now uses the cached list directly when the sample size
   equals the full dataset, so both tables measure the same thing the
   same way.
2. **The mlxtend-authenticity question.** Earlier drafts of this
   codebase (not in this delivery) used a genuine from-scratch Apriori/
   FP-Growth reimplementation before migrating to real mlxtend calls.
   This was checked directly: `apriori.py` and `fpgrowth.py` in this
   delivery are confirmed thin wrappers around real
   `mlxtend.frequent_patterns`, using `TransactionEncoder` correctly
   (see `mlxtend_helpers.py`). The paper's specific technical claims
   (mlxtend version 0.25.0, PR #567/#619/#622, the `low_memory=True`
   test) are accurate.
3. **A bug in my own profiling script**, found while actually running
   it: the header-detection logic checked for the string "cumulative"
   in cProfile's output, but the real header says "cumtime". Fixed and
   re-verified by actually running it again.

## RESOLVED: Table IV vs. Table V (was 103 vs. 104 itemsets, Kaggle data)

This is now fixed, for real, not just by reasoning. Your real
`Market_Basket_Optimisation.csv` (7,501 real transactions, 119 real
products) is included in this delivery at
`code/real_data/Market_Basket_Optimisation.csv`, and
`run_real_data_study.py` was actually run against it as part of this
update. Both tables now agree exactly:

```
support=0.02, n=7501: 103 itemsets (support-sensitivity path)
n=7501, support=0.02: 103 itemsets (volume-scaling path)
```

The previous 103-vs-104 mismatch does not reproduce with the current
code and your actual data file -- it most likely came from a different
version of the code, or a different copy of the CSV, being used for
the two tables at different points during the paper's earlier
development. The real, current, verified numbers (median of 3 trials)
are saved in `data/real_table1_support_sensitivity.csv` and
`data/real_table2_volume_scaling.csv` in this delivery, and the real
cost ratio at the hardest tested configuration (n=7,501, support=0.01)
is genuinely 10,421.9x, not whatever value was in the paper before --
replace Table IV, Table V, and the corresponding cost-ratio sentence in
the paper with these verified numbers.

One honest thing worth flagging directly: the real support-sensitivity
ratio (apriori_time / fpgrowth_time) is not perfectly monotonic --
0.30 (support=0.05) -> 0.47 (0.03) -> 0.26 (0.02) -> 0.73 (0.01) -- see
`figures/figure8_real_data_support_sensitivity.png`. The paper's
current text describes a smoothly "narrowing gap toward parity" as
support decreases; the real data is closer to that story on average
but genuinely non-monotonic at support=0.02 specifically, at only
n=7,501 real transactions where a single timing measurement (even at
median-of-3) can still be noisy. Consider re-running with more trials
before finalizing this claim's exact wording, or softening "narrows"
to allow for this real variability.

## New code added, addressing specific reviewer points

- **`env_info.py`** (point 5: hardware/software rigor) -- captures
  CPU, RAM, OS, and exact Python/NumPy/pandas/mlxtend versions. Run
  this once per machine and keep the output alongside any results.
- **`profile_mechanism.py`** (point 4: "profile both functions") --
  direct cProfile decomposition of Apriori and FP-Growth internals.
  Already run once during this update and found something genuinely
  informative: TransactionEncoder conversion overhead is ~40% of
  Apriori's total measured time but only ~9% of FP-Growth's (since
  FP-Growth's own tree-building work is so much larger in absolute
  terms). Since this overhead is shared and roughly similar in
  absolute size, this means it works AGAINST the "Apriori is faster"
  finding, not for it -- the true algorithmic gap, once this shared
  overhead is excluded, is likely larger than currently reported, not
  an artifact explaining it away.
- **`benchmark_alternative_implementation.py`** (point 3: benchmark a
  second implementation) -- compares mlxtend's Apriori against the
  independent `efficient-apriori` PyPI package. Already run once: at
  every volume tested, mlxtend was faster (ratio 0.50-0.92x) and both
  implementations found identical itemsets. This is a real, if narrow,
  result: it doesn't test FP-Growth against alternatives (efficient-
  apriori doesn't implement FP-Growth), but it does directly test
  whether mlxtend's Apriori is unusually fast among Apriori
  implementations specifically.

## Fixes to existing code

- **`run_experiments.py`**: previously measured each timing point once;
  now uses median-of-3 trials with standard deviation reported for
  every point (matching the methodology already used in the real-data
  scripts), directly addressing the reviewer's rigor concern. The
  confusing "Speedup" column (values below 1.0 meant Apriori was
  faster, an inversion of what "speedup" normally implies) is renamed
  to `apriori_fpgrowth_ratio` with an explicit printed clarification of
  direction. The vague "more than a dozen configurations" is replaced
  with the exact count (32: 8 volumes x 4 supports).
- **`data_sim.py`**: the catalog was 30 items, correctly flagged as
  unrealistic for retail (compare: 119 products in the real Kaggle
  dataset used elsewhere in this study, 4,000+ in the real UCI
  dataset). Expanded to 20 category clusters; because realistic items
  like onion, milk, and sugar deliberately recur across multiple
  clusters (modeling how real staple ingredients appear in many meal
  contexts), the resulting distinct catalog is 79 items -- verified by
  actually running the file, not asserted from the cluster list's
  nominal size, so the docstring states 79, not a rounder but
  inaccurate number.

## Folder structure

```
elastic_pipeline/
├── requirements.txt
├── run_all.py                          <- run this
├── code/
│   ├── apriori.py                        thin wrapper around real mlxtend
│   ├── fpgrowth.py                        thin wrapper around real mlxtend
│   ├── mlxtend_helpers.py                 shared TransactionEncoder conversion
│   ├── data_sim.py                        synthetic retail data (79-item catalog)
│   ├── cost_model.py                      real AWS Lambda/EMR pricing formulas
│   ├── distributed_fpgrowth.py            analytical k-worker cluster simulation
│   ├── run_experiments.py                 main synthetic study (Tables 1-4, Figs 1-4)
│   ├── run_real_data_study.py             Kaggle real-data validation (Table IV/V bug still open)
│   ├── fetch_real_data_locally.py         UCI full-scale extension (Table VI/VII: FIXED)
│   ├── env_info.py                        NEW: hardware/software capture
│   ├── profile_mechanism.py               NEW: cProfile decomposition
│   └── benchmark_alternative_implementation.py   NEW: mlxtend vs. efficient-apriori
├── data/                                 (CSV/JSON results land here)
└── figures/                              (PNG figures land here)
```

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

Steps that need data you'll need to provide yourself (the Kaggle CSV,
or network access to UCI) are skipped automatically with a clear
message rather than crashing the whole run -- this was verified
directly by actually running `run_all.py` in an environment missing
both.

### For the real-data validation (Table IV/V) -- already done for you

Your real `Market_Basket_Optimisation.csv` is already included at
`code/real_data/Market_Basket_Optimisation.csv` in this delivery, and
`run_real_data_study.py` has already been run against it as part of
this update (see the RESOLVED section above and `data/real_table*.csv`
for the verified results). Re-run it yourself anytime with
`cd code && python run_real_data_study.py` to confirm independently.

### For the full-scale UCI extension (Table VI/VII, now fixed)

Needs network access to `archive.ics.uci.edu` (blocked in the sandbox
this update was built in, so the fix above could not be re-verified
against real UCI data end-to-end -- only the logic was verified
directly). Run `cd code && python fetch_real_data_locally.py` on a
machine with internet access. The first successful run caches the
541,909-row download locally (`uci_online_retail_transactions_cache.pkl`),
so subsequent runs, including any further debugging, don't need to
re-download.

## Still open (not code -- needs paper text changes or your input)

- **Points 1, 2, 8** (AWS deployment framing): the paper already
  discloses that EMR is modeled analytically and Lambda's 1024MB is
  assumed, not deployed -- but doesn't address the reviewer's sharper
  points (Lambda CPU scales with memory; the cost "finding" is true by
  construction given the 60-second billing floor). Needs paper-text
  reframing, since actual AWS deployment isn't something this update
  can do.
- **Point 3** (title): still needs to drop "big data" and clarify the
  finding is about mlxtend specifically.
- **Point 6** (paper-text issues): the false "only configuration where
  FP-Growth wins" claim (contradicted by the paper's own Table VI), the
  92.5-fold figure that doesn't match Table III, and adding FP-Growth's
  absolute times to the relevant paper tables all still need direct
  edits to the paper.
- **Point 7** (missing citations): Savasere et al. 1995 (Partition/SON)
  and mlxtend PR #634 still need adding to the bibliography.
- **Point 10** (related work): PyWren/Lithops, serverless Spark, and
  FIMI workshop benchmarking references still need adding.
- **Point 11** (code repository link): needs your actual repo URL.
- **Point 12** (prose): abstract length and repeated phrasing still
  need editing.
