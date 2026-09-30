# An Elastic Cloud Pipeline for Small-to-Big Data Transition in Frequent Pattern Mining

Code for the paper of the same name, submitted to IEEE ICAST 2026 (Paper ID 39).


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