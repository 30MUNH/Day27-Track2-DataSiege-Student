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

print("Testing Rolling Z-score for Lineage Run...")
lr_events = []
for i, (ev, label) in enumerate(zip(events, labels)):
    if ev["type"] == "lineage_run":
        ref = ev["payload"].get("run_id")
        gt = gt_by_key.get(("lineage_run", ref), {})
        lr_events.append({
            "dur": gt["lineage_duration_ms"],
            "faulty": label["is_faulty"],
            "seq": i,
            "label": label.get("fault_key")
        })

for window in range(2, 30):
    clean_zs = []
    faulty_zs = []
    
    for i in range(len(lr_events)):
        curr = lr_events[i]
        
        # history up to i
        hist = [e["dur"] for e in lr_events[max(0, i-window):i]]
        
        if len(hist) > 1:
            m = sum(hist)/len(hist)
            s = (sum((x-m)**2 for x in hist)/len(hist))**0.5
            if s > 0:
                z = abs(curr["dur"] - m) / s
                if curr["faulty"] and curr["label"] == "runtime_anomaly":
                    faulty_zs.append(z)
                elif not curr["faulty"]:
                    clean_zs.append(z)
                    
    if faulty_zs and clean_zs and min(faulty_zs) > max(clean_zs):
        print(f"Window {window} WORKS! min faulty z={min(faulty_zs):.4f}, max clean z={max(clean_zs):.4f}")
    elif faulty_zs and clean_zs:
        print(f"Window {window}: min faulty z={min(faulty_zs):.4f}, max clean z={max(clean_zs):.4f}")

