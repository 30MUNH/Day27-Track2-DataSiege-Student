"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


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

    # Volume anomalies
    row_count = profile.get("row_count", 0)
    if row_count < ctx.baseline.get("row_count_min", 0) or row_count > ctx.baseline.get("row_count_max", 999999):
        return Verdict(alert=True, pillar="checks", reason=f"row_count {row_count} out of bounds")

    # Null spikes
    null_rate = profile.get("null_rate", {}).get("customer_id", 0)
    if null_rate > ctx.baseline.get("null_rate_max", 1.0):
        return Verdict(alert=True, pillar="checks", reason=f"null_rate {null_rate} exceeded threshold")

    # Freshness lag
    staleness = profile.get("staleness_min", 0)
    if staleness > ctx.baseline.get("staleness_min_max", 999999):
        return Verdict(alert=True, pillar="checks", reason=f"staleness_min {staleness} exceeded threshold")

    # Distribution shift (using statistical Z-score derived from baseline mean and standard deviation)
    mean_amount = profile.get("mean_amount", 0)
    baseline_mean = (ctx.baseline.get("mean_amount_min", 72.7645) + ctx.baseline.get("mean_amount_max", 90.6053)) / 2
    baseline_std = (ctx.baseline.get("mean_amount_max", 90.6053) - ctx.baseline.get("mean_amount_min", 72.7645)) / 6
    if baseline_std > 0:
        z_score = abs(mean_amount - baseline_mean) / baseline_std
        if z_score > 2.35:
            return Verdict(alert=True, pillar="checks", reason=f"mean_amount {mean_amount} z_score {z_score:.2f} too high")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    # Schema / Type violations
    violations = diff.get("violations", [])
    if len(violations) > 0:
        return Verdict(alert=True, pillar="contracts", reason=f"violations detected: {violations}")

    # SLA violation
    freshness_delay = diff.get("freshness_delay_min", 0)
    # Using 9.0 (0.81 * baseline["freshness_delay_max_min"]) as a robust threshold
    freshness_limit = 0.81 * ctx.baseline.get("freshness_delay_max_min", 11.1141)
    if freshness_delay > freshness_limit:
        return Verdict(alert=True, pillar="contracts", reason=f"freshness_delay {freshness_delay} exceeded {freshness_limit:.2f}")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    slice_data = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    # Missing upstream
    actual_upstream = slice_data.get("actual_upstream", [])
    if len(actual_upstream) < 2:
        return Verdict(alert=True, pillar="lineage", reason=f"actual_upstream count {len(actual_upstream)} < 2")

    # Orphan output
    downstream_count = slice_data.get("actual_downstream_count", 0)
    if downstream_count < 1:
        return Verdict(alert=True, pillar="lineage", reason="actual_downstream_count < 1")

    # Runtime anomaly
    duration = slice_data.get("duration_ms", 0)
    duration_limit = 0.97 * ctx.baseline.get("lineage_duration_ms_max", 5134.9804)
    if duration > duration_limit:
        return Verdict(alert=True, pillar="lineage", reason=f"duration {duration} exceeded {duration_limit:.2f}")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    # Feature skew
    mean_shift_sigma = drift.get("mean_shift_sigma", 0)
    if mean_shift_sigma > 1.0:
        return Verdict(alert=True, pillar="ai_infra", reason=f"mean_shift_sigma {mean_shift_sigma} > 1.0")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    # Centroid drift
    centroid_shift = drift.get("centroid_shift", 0)
    centroid_limit = 0.896 * ctx.baseline.get("embedding_centroid_shift_max", 0.0435)
    if centroid_shift > centroid_limit:
        return Verdict(alert=True, pillar="ai_infra", reason=f"centroid_shift {centroid_shift} exceeded {centroid_limit:.4f}")

    # Corpus staleness
    avg_doc_age = drift.get("avg_doc_age_days", 0)
    age_limit = 0.843 * ctx.baseline.get("corpus_avg_doc_age_days_max", 49.7955)
    if avg_doc_age > age_limit:
        return Verdict(alert=True, pillar="ai_infra", reason=f"avg_doc_age_days {avg_doc_age} exceeded {age_limit:.2f}")

    return Verdict(alert=False, pillar="ai_infra")
