#python - model_name fg_rebuttal <<'PY'
import csv, re, sys
from collections import defaultdict
from pathlib import Path

model, tag = sys.argv[1], sys.argv[2]
root = Path("runs")
paths = sorted(root.glob(f"{model}/seed*_{tag}/results.csv"))

pair = defaultdict(lambda: {"target": 0, "reliability": 0, "brec": 0, "seeds": set()})

for path in paths:
    seed_match = re.search(r"seed(\d+)", path.parent.name)
    seed = int(seed_match.group(1)) if seed_match else path.parent.name
    with path.open() as f:
        for row in csv.DictReader(f):
            i = int(row["pair_index"])
            passed = row.get("passed", "False") == "True"
            pair[i]["seeds"].add(seed)
            if row["trial_type"] == "target_surface_vs_free" and passed:
                pair[i]["target"] += 1
            if row["trial_type"] == "reliability_surface_vs_surface" and passed:
                pair[i]["reliability"] += 1

for i, d in pair.items():
    d["brec"] = min(d["target"], d["reliability"])

print("pair_index,target_successes/4,reliability_successes/4,brec_successes/4,consistently_solved")
consistent = 0
for i in sorted(pair):
    d = pair[i]
    ok = d["target"] == 4 and d["reliability"] == 4
    consistent += int(ok)
    print(f"{i},{d['target']}/4,{d['reliability']}/4,{d['brec']}/4,{ok}")

print(f"\nUnique pairs consistently solved: {consistent}/{len(pair)}")
#