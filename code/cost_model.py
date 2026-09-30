"""
cost_model.py
---------------
Cost model for the serverless (AWS Lambda) and cluster (AWS EMR)
architectures, using real, publicly published AWS pricing verified
directly against AWS's own pricing pages (accessed 2026; see paper
Section III for exact figures and citation). This is real pricing
data applied to REAL measured algorithm runtimes (from apriori.py,
fpgrowth.py, distributed_fpgrowth.py), not a hypothetical model.

Pricing constants (verified against aws.amazon.com/lambda/pricing and
aws.amazon.com/emr/pricing, and third-party EMR-on-EC2 rate summaries
citing the same AWS source, as of the access date in the paper):
  Lambda:  $0.20 per 1M requests; $0.0000166667 per GB-second compute;
           free tier 1M requests + 400,000 GB-s/month (not applied
           here, to model a retailer already past free-tier volume).
  EMR:     m5.xlarge = $0.192/hr (EC2) + $0.048/hr (EMR surcharge)
           = $0.240/hr per node (4 vCPU, 16 GiB RAM each).
"""

LAMBDA_PRICE_PER_REQUEST = 0.20 / 1_000_000
LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667
LAMBDA_MEMORY_GB = 1.0  # assume a 1024MB Lambda, a reasonable default for this workload

EMR_NODE_PRICE_PER_HOUR = 0.240  # m5.xlarge, EC2 + EMR surcharge combined
EMR_NODE_VCPUS = 4


def serverless_cost(runtime_seconds, n_invocations=1, memory_gb=LAMBDA_MEMORY_GB):
    """Cost of running the mining job as `n_invocations` Lambda
    invocations totaling `runtime_seconds` of compute time.
    """
    gb_seconds = memory_gb * runtime_seconds
    compute_cost = gb_seconds * LAMBDA_PRICE_PER_GB_SECOND
    request_cost = n_invocations * LAMBDA_PRICE_PER_REQUEST
    return compute_cost + request_cost


def cluster_cost(wall_clock_seconds, n_nodes):
    """Cost of running the mining job on an n_nodes EMR cluster for
    wall_clock_seconds, billed per-second with AWS's real EMR per-second
    billing (one-minute minimum, applied here).
    """
    billed_seconds = max(60, wall_clock_seconds)  # one-minute EMR minimum
    hours = billed_seconds / 3600.0
    return n_nodes * EMR_NODE_PRICE_PER_HOUR * hours


def find_crossover(volumes, serverless_costs, cluster_costs):
    """Finds the transaction volume at which cluster_cost first becomes
    cheaper than serverless_cost, by linear interpolation between the
    two bracketing measured points. Returns None if no crossover is
    observed in the measured range.
    """
    for i in range(1, len(volumes)):
        if serverless_costs[i - 1] <= cluster_costs[i - 1] and serverless_costs[i] > cluster_costs[i]:
            # crossover between i-1 and i; linear interpolation
            x0, x1 = volumes[i - 1], volumes[i]
            d0 = serverless_costs[i - 1] - cluster_costs[i - 1]
            d1 = serverless_costs[i] - cluster_costs[i]
            if d1 != d0:
                frac = -d0 / (d1 - d0)
                return x0 + frac * (x1 - x0)
            return x1
    return None
