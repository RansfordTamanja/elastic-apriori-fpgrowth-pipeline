"""
data_sim.py
------------
Generates synthetic retail transaction data at varying scale, used
throughout this study to benchmark Apriori vs. FP-Growth vs. simulated
distributed FP-Growth as transaction volume grows from small-retailer
to large-retailer scale.

Transactions are drawn from a fixed catalog of items with realistic
co-occurrence structure (a set of "association clusters", e.g. bread
tends to co-occur with butter and milk), rather than pure random
sampling, so that non-trivial frequent itemsets actually exist in the
generated data at every scale.

CATALOG SIZE: an earlier version of this file used a 30-item catalog,
correctly flagged as unrealistic for retail -- both real datasets used
elsewhere in this study have far more distinct products (119 in the
Kaggle Market Basket dataset, over 4,000 in the full UCI Online Retail
dataset). The catalog here is expanded to 20 realistic category
clusters; because several items (e.g. onion, milk, sugar) deliberately
appear in more than one cluster, matching how real staple ingredients
recur across different meal/purchase occasions, the resulting distinct
catalog is 79 items (verified directly by running this file, not
assumed from the cluster list's nominal size), roughly 2.6x the
original catalog and broadly comparable to a small convenience store's
core SKU range. This is smaller than the Kaggle dataset's 119 products
and far smaller than UCI's 4,000+, a real, disclosed limitation the
synthetic study should be read alongside its two real-data validations
(Sections V-D and V-E of the paper) rather than in isolation.
"""
import numpy as np


CLUSTERS = [
    ["bread", "butter", "milk", "eggs", "jam", "honey"],
    ["rice", "beans", "lentils", "oil", "onion", "tomato", "garlic"],
    ["chicken", "pepper", "onion", "salt", "garlic", "ginger", "curry_powder"],
    ["soap", "detergent", "tissue", "sponge", "bleach"],
    ["cola", "chips", "biscuit", "candy", "popcorn"],
    ["flour", "sugar", "eggs", "butter", "baking_powder", "vanilla"],
    ["fish", "tomato", "pepper", "onion", "lemon", "parsley"],
    ["diapers", "yogurt", "juice", "baby_formula", "baby_wipes"],
    ["pasta", "tomato_sauce", "parmesan", "olive_oil", "basil"],
    ["coffee", "sugar", "milk", "creamer", "biscuit"],
    ["tea", "sugar", "milk", "honey", "lemon"],
    ["beef", "onion", "pepper", "potato", "carrot", "beef_stock"],
    ["shampoo", "conditioner", "body_wash", "toothpaste", "toothbrush"],
    ["cereal", "milk", "banana", "honey"],
    ["salad_greens", "tomato", "cucumber", "olive_oil", "vinegar", "feta"],
    ["beer", "chips", "peanuts", "pretzels"],
    ["water", "sports_drink", "energy_bar", "banana"],
    ["yogurt", "granola", "berries", "honey"],
    ["cheese", "crackers", "grapes", "wine"],
    ["laundry_pods", "fabric_softener", "dryer_sheets", "stain_remover"],
]

# CATALOG is derived directly from CLUSTERS (every item that appears in
# at least one cluster), rather than maintained as a separately-typed
# list, so the two can never silently drift out of sync.
CATALOG = sorted(set(item for cluster in CLUSTERS for item in cluster))


def generate_transactions(n_transactions, seed=0, avg_basket_size=6):
    """Returns a list of n_transactions lists of item names."""
    rng = np.random.RandomState(seed)
    transactions = []
    for _ in range(n_transactions):
        basket = set()
        # Each transaction draws from 1-2 clusters plus some noise items,
        # modeling realistic co-purchase behavior rather than pure noise.
        n_clusters_this_basket = rng.choice([1, 2], p=[0.65, 0.35])
        chosen_clusters = rng.choice(len(CLUSTERS), size=n_clusters_this_basket, replace=False)
        for c_idx in chosen_clusters:
            cluster = CLUSTERS[c_idx]
            n_from_cluster = rng.randint(2, len(cluster) + 1)
            items = rng.choice(cluster, size=min(n_from_cluster, len(cluster)), replace=False)
            basket.update(items)
        # Add random noise items to vary basket size around avg_basket_size
        n_noise = max(0, int(rng.normal(avg_basket_size - len(basket), 1.5)))
        noise_items = rng.choice(CATALOG, size=min(n_noise, len(CATALOG)), replace=False)
        basket.update(noise_items)
        if len(basket) == 0:
            basket.add(rng.choice(CATALOG))
        transactions.append(sorted(basket))
    return transactions


if __name__ == "__main__":
    for n in [100, 1000, 10000]:
        txns = generate_transactions(n, seed=0)
        avg_len = np.mean([len(t) for t in txns])
        print(f"n_transactions={n}: avg basket size={avg_len:.2f}, "
              f"example basket={txns[0]}")
