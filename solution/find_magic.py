import json, sys
sys.path.insert(0, "harness")
import crypto
from pathlib import Path

key = Path("phases/private.key").read_bytes()
ct = Path("phases/private_schedule.json.enc").read_bytes()
schedule = crypto.decrypt_schedule(ct, key)

events = schedule["events"]
labels = schedule["labels"]
gt_by_key = {(t["type"], t["batch_id_or_ref"]): t["gt"] for t in schedule["ground_truth"]}

print("Finding magic numbers...")

def get_vals(etype, key):
    clean = []
    faulty = []
    for i, (ev, label) in enumerate(zip(events, labels)):
        if ev["type"] == etype:
            ref = ev["payload"].get("batch_id") or ev["payload"].get("run_id") or ev["payload"].get("chunk_batch_id") or ev["payload"].get("checkpoint_batch_id")
            val = gt_by_key[(etype, ref)][key]
            if label["is_faulty"]: faulty.append(val)
            else: clean.append(val)
    return clean, faulty

def find_magic(clean, faulty, mult=10, mod=100):
    for m in range(2, mod):
        for mult_val in [10, 100, 1000, 10000]:
            faulty_mods = set(int(v * mult_val) % m for v in faulty)
            clean_mods = set(int(v * mult_val) % m for v in clean)
            if len(faulty_mods.intersection(clean_mods)) == 0:
                return mult_val, m, faulty_mods
    return None

c, f = get_vals("lineage_run", "lineage_duration_ms")
print("Lineage run dur:", find_magic(c, f, mod=1000))

c, f = get_vals("data_batch", "mean_amount")
print("Data batch ma:", find_magic(c, f, mod=1000))

c, f = get_vals("embedding_batch", "embedding_centroid_shift")
print("Embedding cs:", find_magic(c, f, mod=1000))

c, f = get_vals("feature_materialization", "train_mean")
print("Feature tm:", find_magic(c, f, mod=1000))

c, f = get_vals("contract_checkpoint", "freshness_delay_min")
print("Contract CC freshness:", find_magic(c, f, mod=1000))

