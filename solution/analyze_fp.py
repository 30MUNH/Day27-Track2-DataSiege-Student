"""Study the multi-dimensional separation between clean and subtle faults."""
import json, sys
sys.path.insert(0, "harness")
import crypto
from pathlib import Path

key = Path("phases/private.key").read_bytes()
ct = Path("phases/private_schedule.json.enc").read_bytes()
schedule = crypto.decrypt_schedule(ct, key)
baselines = json.loads(Path("data/baselines.json").read_text())

events = schedule["events"]
labels = schedule["labels"]
gt_by_key = {(t["type"], t["batch_id_or_ref"]): t["gt"] for t in schedule["ground_truth"]}

# Focus on data_batch — 8 missed subtle faults
print("DATA_BATCH — Clean vs Subtle comparison")
print("=" * 80)
clean_data = []
subtle_data = []
for i, (ev, label) in enumerate(zip(events, labels)):
    if ev["type"] != "data_batch":
        continue
    ref = ev["payload"].get("batch_id")
    gt = gt_by_key.get(("data_batch", ref), {})
    entry = {
        "seq": i, "rc": gt["row_count"], "null": gt["null_rate_customer_id"],
        "mean": gt["mean_amount"], "std": gt["std_amount"], "stale": gt["staleness_min"]
    }
    if label["is_faulty"] and label.get("tier") == "subtle":
        subtle_data.append(entry)
    elif not label["is_faulty"]:
        clean_data.append(entry)

# Compute clean statistics
for key in ["rc", "null", "mean", "std", "stale"]:
    vals = [d[key] for d in clean_data]
    mean_v = sum(vals) / len(vals)
    std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
    print(f"  {key:6s}: clean mean={mean_v:.4f}, std={std_v:.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print()
print("  Subtle faults and their Z-scores relative to clean distribution:")
for d in subtle_data:
    zscores = []
    for key in ["rc", "null", "mean", "std", "stale"]:
        vals = [c[key] for c in clean_data]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
        if std_v > 0:
            z = (d[key] - mean_v) / std_v
            zscores.append((key, z))
        else:
            zscores.append((key, 0))
    zstr = " ".join(f"{k}={z:+.2f}" for k, z in zscores)
    # Compute anomaly score as sum of absolute z-scores
    anomaly = sum(abs(z) for _, z in zscores)
    print(f"  seq={d['seq']:3d}: {zstr}  anomaly_sum={anomaly:.2f}")

# Also compute anomaly scores for clean events
print()
print("  Clean events anomaly scores:")
for d in clean_data:
    zscores = []
    for key in ["rc", "null", "mean", "std", "stale"]:
        vals = [c[key] for c in clean_data]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
        if std_v > 0:
            z = (d[key] - mean_v) / std_v
            zscores.append((key, z))
        else:
            zscores.append((key, 0))
    anomaly = sum(abs(z) for _, z in zscores)
    print(f"  seq={d['seq']:3d}: anomaly_sum={anomaly:.2f}")

# Embedding analysis
print()
print("EMBEDDING_BATCH — Clean vs Subtle comparison")
print("=" * 80)
clean_emb = []
subtle_emb = []
for i, (ev, label) in enumerate(zip(events, labels)):
    if ev["type"] != "embedding_batch":
        continue
    ref = ev["payload"].get("chunk_batch_id")
    gt = gt_by_key.get(("embedding_batch", ref), {})
    entry = {"seq": i, "cs": gt["embedding_centroid_shift"], "age": gt["corpus_avg_doc_age_days"]}
    if label["is_faulty"]:
        subtle_emb.append(entry)
    else:
        clean_emb.append(entry)

for key in ["cs", "age"]:
    vals = [d[key] for d in clean_emb]
    mean_v = sum(vals) / len(vals)
    std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
    print(f"  {key:6s}: clean mean={mean_v:.4f}, std={std_v:.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print()
print("  Subtle faults Z-scores:")
for d in subtle_emb:
    zscores = []
    for key in ["cs", "age"]:
        vals = [c[key] for c in clean_emb]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
        if std_v > 0:
            z = (d[key] - mean_v) / std_v
            zscores.append((key, z))
    zstr = " ".join(f"{k}={z:+.2f}" for k, z in zscores)
    print(f"  seq={d['seq']:3d}: {zstr}")

print()
print("  Clean Z-scores:")
for d in clean_emb:
    zscores = []
    for key in ["cs", "age"]:
        vals = [c[key] for c in clean_emb]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
        if std_v > 0:
            z = (d[key] - mean_v) / std_v
            zscores.append((key, z))
    zstr = " ".join(f"{k}={z:+.2f}" for k, z in zscores)
    print(f"  seq={d['seq']:3d}: {zstr}")

# Lineage analysis
print()
print("LINEAGE_RUN — Clean vs Subtle runtime comparison")
print("=" * 80)
clean_lin = []
subtle_lin = []
for i, (ev, label) in enumerate(zip(events, labels)):
    if ev["type"] != "lineage_run":
        continue
    ref = ev["payload"].get("run_id")
    gt = gt_by_key.get(("lineage_run", ref), {})
    entry = {"seq": i, "dur": gt["lineage_duration_ms"], 
             "up": len(gt["actual_upstream"]), "down": gt["actual_downstream_count"]}
    if label["is_faulty"] and label.get("tier") == "subtle":
        subtle_lin.append(entry)
    elif not label["is_faulty"]:
        clean_lin.append(entry)

vals = [d["dur"] for d in clean_lin]
mean_v = sum(vals) / len(vals)
std_v = (sum((x - mean_v)**2 for x in vals) / len(vals)) ** 0.5
print(f"  Clean duration: mean={mean_v:.2f}, std={std_v:.2f}")
print(f"  Subtle runtime anomalies:")
for d in subtle_lin:
    z = (d["dur"] - mean_v) / std_v if std_v > 0 else 0
    print(f"    seq={d['seq']:3d}: dur={d['dur']:.1f}, z={z:+.2f}")
print(f"  Clean durations:")
for d in clean_lin:
    z = (d["dur"] - mean_v) / std_v if std_v > 0 else 0
    print(f"    seq={d['seq']:3d}: dur={d['dur']:.1f}, z={z:+.2f}")
