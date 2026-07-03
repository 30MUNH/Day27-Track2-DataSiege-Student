# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**
The hardest faults to catch were the subtle-magnitude faults that fell within the static ±3σ baseline boundaries:
1. **Subtle `distribution_shift`** (e.g., Seq 95 in the public phase with a `mean_amount` of 88.91, below the baseline max of 90.6053).
2. **Subtle `embedding_drift`** (e.g., Seq 24 in the public phase with a `centroid_shift` of 0.0400, below the baseline max of 0.0435).
3. **Subtle `corpus_staleness`** (e.g., Seq 19 in the public phase with an `avg_doc_age_days` of 48.3, below the baseline max of 49.7955).

These faults were difficult because simple static threshold checks against the baseline limits result in false negatives. To detect them, we had to apply statistical analysis: calculating the Z-score of the mean amount relative to the baseline distribution and calibrating tighter thresholds (e.g., 2.35σ limits) derived from the observed clean-stream metrics.

**What would you change about your cost/coverage tradeoff, if you had another pass?**
Currently, we run full checks on all events. This costs 240.0 credits (over the 220.0 budget by 9.09%), causing a minor penalty of 1.82 points but guaranteeing a 100% True Positive Rate (TPR) and 0% False Positive Rate (FPR) for a net score of 48.18.

If we had another pass, we could implement a conditional skipping strategy:
1. **Skip checking lower-risk event types** (e.g., `embedding_batch` had only a 12.5% fault rate in public) or sample them periodically.
2. **Implement stateful alerts**: If we detect a lineage run or schema breach, we could predict downstream impacts or skip redundant checks for the next step of the pipeline.

However, since a single missed fault drops the score by 1.51 points while saving 2 credits only saves 0.18 points in cost overage (an 8.3x difference), full coverage remains the mathematically optimal choice under the current scoring weights.
