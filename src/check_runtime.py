import json, numpy as np
d = json.load(open("exp/supplementary_results.json"))
r = d["results"]
seeds = [str(s) for s in range(10)]
print("Runtime comparison FITI vs FLSE:")
for ds in sorted(r.keys()):
    fiti_times = [r[ds][s]["fiti"][0].get("time", 0) for s in seeds]
    flse_times = [r[ds][s]["flse"][0].get("time", 0) for s in seeds]
    if any(t > 0 for t in fiti_times) and any(t > 0 for t in flse_times):
        ratio = np.mean(fiti_times) / np.mean(flse_times) if np.mean(flse_times) > 0 else float("inf")
        print(f"  {ds}: FLSE={np.mean(flse_times):.4f}s FITI={np.mean(fiti_times):.4f}s ratio={ratio:.2f}x")
    else:
        print(f"  {ds}: no time data (time field=0)")
