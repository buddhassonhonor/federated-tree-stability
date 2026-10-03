"""
Corrected statistical table for FITI paper .
Computes:
 A: FITI vs FLSE (primary)
 B: FITI vs Centralized CART (reference)
 C: FITI-Forest vs FedRF Std (exploratory)
Rules:
- If paired differences have zero variance -> report "Tie", "Constant+", or "Constant-"
- Holm-Bonferroni correction across 7 datasets per comparison type
- Bootstrap 95% CI on mean difference
- Cohen's d on paired differences (undefined if std=0)
"""
import json
import numpy as np
from scipy.stats import wilcoxon

np.random.seed(42)

def bootstrap_ci(diffs, n_boot=5000, alpha=0.05):
    """95% bootstrap CI for mean of paired differences."""
    means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) 
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi

def cohend_paired(diffs):
    """Cohen's d for paired differences = mean(diffs)/std(diffs)."""
    s = np.std(diffs, ddof=1)
    if s < 1e-10:
        return None  # undefined
    return np.mean(diffs) / s

def holm_bonferroni(pvals, alpha=0.05):
    """Return Holm-Bonferroni adjusted significance decisions."""
    n = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject

def format_pval(p):
    if p < 0.001:
        return "$< 0.001$"
    return f"${p:.3f}$"

def format_ci(lo, hi):
    sign_lo = "+" if lo >= 0 else ""
    sign_hi = "+" if hi >= 0 else ""
    return f"[{sign_lo}{lo:.3f}, {sign_hi}{hi:.3f}]"

def format_d(d):
    if d is None:
        return "undef."
    sign = "+" if d >= 0 else ""
    return f"${sign}{d:.2f}$"

def deterministic_label(mean_d, tol=1e-10):
    """Label deterministic paired differences without assigning Wilcoxon p-values."""
    if abs(mean_d) < tol:
        return "Tie"
    return "Constant+" if mean_d > 0 else "Constant-"

with open("exp/supplementary_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
seeds = [str(s) for s in range(10)]

datasets = sorted(results.keys())

# ---- Comparison A: FITI vs FLSE ----
print("\n% === Comparison A: FITI vs FLSE (primary) ===")
pvals_A = []
rows_A = []
for ds in datasets:
    fiti = [results[ds][s]["fiti"][0]["ood_accuracy"] for s in seeds]
    flse = [results[ds][s]["flse"][0]["ood_accuracy"] for s in seeds]
    diffs = np.array(fiti) - np.array(flse)
    mean_d = np.mean(diffs)
    lo, hi = bootstrap_ci(diffs)
    d = cohend_paired(diffs)
    if np.std(diffs, ddof=1) < 1e-10:
        p = None
    else:
        _, p = wilcoxon(fiti, flse, alternative="two-sided")
    pvals_A.append(p if p is not None else 1.0)
    rows_A.append((ds, mean_d, lo, hi, p, d))

reject_A = holm_bonferroni(pvals_A)

print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("Dataset & Mean Diff & 95\\% CI & Wilcoxon $p$ & HB & Cohen's $d_p$ \\\\")
print("\\midrule")
for i, (ds, mean_d, lo, hi, p, d) in enumerate(rows_A):
    sign = "+" if mean_d >= 0 else ""
    mean_str = f"${sign}{mean_d:.3f}$"
    ci_str = format_ci(lo, hi)
    if p is None:
        p_str = deterministic_label(mean_d)
        hb_str = "---"
    else:
        p_str = format_pval(p)
        hb_str = "\\checkmark" if reject_A[i] else "n.s."
    d_str = format_d(d)
    ds_clean = ds.replace("_", " ")
    print(f"{ds_clean} & {mean_str} & {ci_str} & {p_str} & {hb_str} & {d_str} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# ---- Comparison B: FITI vs Centralized CART ----
print("\n% === Comparison B: FITI vs Centralized CART (reference) ===")
pvals_B = []
rows_B = []
for ds in datasets:
    fiti = [results[ds][s]["fiti"][0]["ood_accuracy"] for s in seeds]
    cart = [results[ds][s]["tuned_cart"][0]["ood_accuracy"] for s in seeds]
    diffs = np.array(fiti) - np.array(cart)
    mean_d = np.mean(diffs)
    lo, hi = bootstrap_ci(diffs)
    d = cohend_paired(diffs)
    if np.std(diffs, ddof=1) < 1e-10:
        p = None
    else:
        _, p = wilcoxon(fiti, cart, alternative="two-sided")
    pvals_B.append(p if p is not None else 1.0)
    rows_B.append((ds, mean_d, lo, hi, p, d))

reject_B = holm_bonferroni(pvals_B)

print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("Dataset & Mean Diff & 95\\% CI & Wilcoxon $p$ & HB & Cohen's $d_p$ \\\\")
print("\\midrule")
for i, (ds, mean_d, lo, hi, p, d) in enumerate(rows_B):
    sign = "+" if mean_d >= 0 else ""
    mean_str = f"${sign}{mean_d:.3f}$"
    ci_str = format_ci(lo, hi)
    if p is None:
        p_str = deterministic_label(mean_d)
        hb_str = "---"
    else:
        p_str = format_pval(p)
        hb_str = "\\checkmark" if reject_B[i] else "n.s."
    d_str = format_d(d)
    ds_clean = ds.replace("_", " ")
    print(f"{ds_clean} & {mean_str} & {ci_str} & {p_str} & {hb_str} & {d_str} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# ---- Comparison C: FITI-Forest vs FedRF Std ----
print("\n% === Comparison C: FITI-Forest vs FedRF Std (exploratory) ===")
pvals_C = []
rows_C = []
for ds in datasets:
    fi_f = [results[ds][s]["fedrf_invariant"][0]["ood_accuracy"] for s in seeds]
    fed_rf = [results[ds][s]["fedrf_standard"][0]["ood_accuracy"] for s in seeds]
    diffs = np.array(fi_f) - np.array(fed_rf)
    mean_d = np.mean(diffs)
    lo, hi = bootstrap_ci(diffs)
    d = cohend_paired(diffs)
    if np.std(diffs, ddof=1) < 1e-10:
        p = None
    else:
        _, p = wilcoxon(fi_f, fed_rf, alternative="two-sided")
    pvals_C.append(p if p is not None else 1.0)
    rows_C.append((ds, mean_d, lo, hi, p, d))

reject_C = holm_bonferroni(pvals_C)

print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("Dataset & Mean Diff & 95\\% CI & Wilcoxon $p$ & HB & Cohen's $d_p$ \\\\")
print("\\midrule")
for i, (ds, mean_d, lo, hi, p, d) in enumerate(rows_C):
    sign = "+" if mean_d >= 0 else ""
    mean_str = f"${sign}{mean_d:.3f}$"
    ci_str = format_ci(lo, hi)
    if p is None:
        p_str = deterministic_label(mean_d)
        hb_str = "---"
    else:
        p_str = format_pval(p)
        hb_str = "\\checkmark" if reject_C[i] else "n.s."
    d_str = format_d(d)
    ds_clean = ds.replace("_", " ")
    print(f"{ds_clean} & {mean_str} & {ci_str} & {p_str} & {hb_str} & {d_str} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# Summary for paper text
print("\n% === Summary for paper narrative ===")
print("% Comparison A significant after HB:", [datasets[i].replace('_',' ') for i in range(7) if reject_A[i]])
print("% Comparison B significant after HB:", [datasets[i].replace('_',' ') for i in range(7) if reject_B[i]])
print("% Comparison C significant after HB:", [datasets[i].replace('_',' ') for i in range(7) if reject_C[i]])
