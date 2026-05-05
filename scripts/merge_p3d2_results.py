#!/usr/bin/env python3
import json
import os

# Load backup with T=4, T=8
with open("results/p3_d2_results.json.backup") as f:
    backup = json.load(f)

# Load new results with T=1, T=2  
all_results = []
for T in [1, 2]:
    temp_file = f"results/p3_d2_t{T}_temp.json"
    try:
        with open(temp_file) as f:
            data = json.load(f)
            for r in data.get("results", []):
                r["num_steps"] = T
                all_results.append(r)
        print(f"Loaded T={T}: {len([r for r in all_results if r.get('num_steps') == T])} results")
    except FileNotFoundError:
        print(f"Warning: T={T} temp file not found")

# Get T=4, T=8 from backup
student_old = [r for r in backup["results"] if r["model_name"] == "trained_midblock" and r["num_steps"] in [4, 8]]
teacher_all = [r for r in backup["results"] if r["model_name"] == "teacher_original"]

# Merge
merged = all_results + student_old + teacher_all

# Save
with open("results/p3_d2_results.json", "w") as f:
    json.dump({"results": merged, "experiment_id": "P3-D2"}, f, indent=2)

print(f"✓ Merged {len(merged)} total results:")
for r in sorted(merged, key=lambda x: (x["model_name"], x["num_steps"])):
    correct = r.get("num_correct", 0)
    total = r.get("num_total", 0)
    acc = r.get("accuracy", 0)
    print(f"  {r['model_name']:20} T={r['num_steps']:2}: {correct}/{total} = {acc:.2%}")

# Cleanup
for T in [1, 2]:
    temp_file = f"results/p3_d2_t{T}_temp.json"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Cleaned up {temp_file}")

print("Done!")
