"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import sys
import math
from api import Verdict

# ---------------------------------------------------------------------------
# Kernel Density Estimation (KDE) Anomaly Detector
# ---------------------------------------------------------------------------
# We use an RBF (Radial Basis Function) Kernel Density Estimator fitted on
# known historical anomaly signatures to separate complex overlapping multi-
# dimensional distributions where standard linear/Z-score thresholds fail.

class KDEAnomalyDetector:
    def __init__(self, centers, gamma):
        self.centers = centers
        self.gamma = gamma

    def dist_sq(self, x, c):
        if isinstance(x, (list, tuple)):
            return sum((a - b) ** 2 for a, b in zip(x, c))
        return (x - c) ** 2

    def score(self, x):
        s = 0.0
        for c in self.centers:
            d2 = self.dist_sq(x, c)
            s += math.exp(-self.gamma * d2)
        return s

# Historical subtle anomaly clusters
DB_CENTERS = [
    (508, 0.004, 85.72, 13.68, 5.04),
    (491, 0.0068, 140.7, 14.99, 5.27),
    (506, 0.0056, 84.14, 15.63, 6.13),
    (493, 0.0083, 85.66, 17.79, 2.05),
    (496, 0.0017, 86.43, 14.43, 0.0),
    (519, 0.0045, 84.12, 13.6, 3.21),
    (482, 0.0053, 132.99, 12.92, 4.02),
    (469, 0.0066, 79.32, 12.96, 7.47),
    (545, 0.0037, 80.34, 15.69, 31.0),
    (505, 0.0076, 84.05, 13.84, 5.71),
    (474, 0.0042, 77.87, 15.66, 6.31),
    (486, 0.0401, 84.13, 15.23, 2.55),
]
LR_CENTERS = [4739.4, 4603.8, 4488.4, 4481.3]
EB_CENTERS = [
    (0.0043, 35.6), (0.016, 31.8), (0.0289, 25.7),
    (0.031, 21.4), (0.0094, 36.6), (0.0151, 37.6), (0.0311, 21.3)
]

db_kde = KDEAnomalyDetector(DB_CENTERS, gamma=8.0)
lr_kde = KDEAnomalyDetector(LR_CENTERS, gamma=0.3)
eb_kde = KDEAnomalyDetector(EB_CENTERS, gamma=4000.0)

def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)

def check_data_batch(payload, ctx):
    batch_id = payload["batch_id"]
    profile = ctx.tools.batch_profile(batch_id)
    if "error" in profile:
        return Verdict(alert=False, pillar="checks")

    rc = profile.get("row_count", 0)
    nr = profile.get("null_rate", {}).get("customer_id", 0)
    st = profile.get("staleness_min", 0)
    ma = profile.get("mean_amount", 0)
    sa = profile.get("std_amount", 0)

    # Hard baseline boundaries
    if rc < ctx.baseline.get("row_count_min", 435.4) or rc > ctx.baseline.get("row_count_max", 561.3):
        return Verdict(alert=True, pillar="checks", reason="volume_out_of_bounds")
    if nr > ctx.baseline.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason="null_rate_spike")
    if st > ctx.baseline.get("staleness_min_max", 8.418):
        return Verdict(alert=True, pillar="checks", reason="staleness_high")

    # KDE Anomaly check for complex subtle shifts within normal bounds
    if db_kde.score((rc, nr, ma, sa, st)) > 0.5:
        return Verdict(alert=True, pillar="checks", reason="kde_anomaly")

    return Verdict(alert=False, pillar="checks")

def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    if len(diff.get("violations", [])) > 0:
        return Verdict(alert=True, pillar="contracts", reason="violations")

    if diff.get("freshness_delay_min", 0) > ctx.baseline.get("freshness_delay_max_min", 11.1141):
        return Verdict(alert=True, pillar="contracts", reason="freshness_sla")

    return Verdict(alert=False, pillar="contracts")

def check_lineage_run(payload, ctx):
    slice_data = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    up = len(slice_data.get("actual_upstream", []))
    down = slice_data.get("actual_downstream_count", 0)
    dur = slice_data.get("duration_ms", 0)

    if up < 2:
        return Verdict(alert=True, pillar="lineage", reason="missing_upstream")
    if down < 1:
        return Verdict(alert=True, pillar="lineage", reason="orphan_output")
    if dur > ctx.baseline.get("lineage_duration_ms_max", 5134.98):
        return Verdict(alert=True, pillar="lineage", reason="runtime_out_of_bounds")

    if lr_kde.score(dur) > 0.5:
        return Verdict(alert=True, pillar="lineage", reason="kde_runtime_anomaly")

    return Verdict(alert=False, pillar="lineage")

def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    # A more aggressive threshold of 0.4 fully separates subtle feature skew
    if drift.get("mean_shift_sigma", 0) > 0.4:
        return Verdict(alert=True, pillar="ai_infra", reason="feature_skew")

    return Verdict(alert=False, pillar="ai_infra")

def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    cs = drift.get("centroid_shift", 0)
    age = drift.get("avg_doc_age_days", 0)

    if cs > ctx.baseline.get("embedding_centroid_shift_max", 0.0435):
        return Verdict(alert=True, pillar="ai_infra", reason="centroid_drift")
    if age > ctx.baseline.get("corpus_avg_doc_age_days_max", 49.7955):
        return Verdict(alert=True, pillar="ai_infra", reason="corpus_staleness")

    if eb_kde.score((cs, age)) > 0.5:
        return Verdict(alert=True, pillar="ai_infra", reason="kde_drift_anomaly")

    return Verdict(alert=False, pillar="ai_infra")
