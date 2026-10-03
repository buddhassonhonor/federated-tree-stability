"""
Feature-selection frequency experiment .
Measures which feature types (invariant/spurious/noise) are chosen at:
  - Root node
  - Top-3 tree levels (depth 0, 1, 2)
For FITI vs FLSE on Synthetic and Adult Spurious.
Reports averaged rates over 10 random seeds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import numpy as np
import yaml

# Load config
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg.get("data_dir", "data")
except:
    data_dir = "data"

from run_supplementary_experiments import get_synthetic_data, get_adult_spurious_data

# ---- Synthetic feature indices ----
# 5 invariant (0-4), 5 spurious (5-9), 10 noise (10-19)
SYNTHETIC_INV   = set(range(0, 5))
SYNTHETIC_SPU   = set(range(5, 10))
SYNTHETIC_NOISE = set(range(10, 20))

# ---- Adult Spurious feature indices ----
# Original features: 0-13 (14 real features) + last feature (14) = injected spurious
# The spurious feature is appended as the last column
def adult_spu_types(n_features):
    spu_idx = n_features - 1
    return (set(range(spu_idx)), {spu_idx}, set())  # (inv, spu, noise)

# ---- FLSE (lambda=0) split criterion ----
def compute_gini_gain(left_pos, left_tot, right_pos, right_tot, parent_pos, parent_tot):
    if parent_tot == 0:
        return 0.0
    def gini(pos, tot):
        if tot == 0: return 0.0
        p = pos / tot
        return 1 - p**2 - (1-p)**2
    g_parent = gini(parent_pos, parent_tot)
    g_left = gini(left_pos, left_tot) if left_tot > 0 else 0.0
    g_right = gini(right_pos, right_tot) if right_tot > 0 else 0.0
    return g_parent - (left_tot/parent_tot)*g_left - (right_tot/parent_tot)*g_right

def get_client_counts(train_nodes, feature, threshold, node_indices):
    """Get per-client left/right class counts for a split."""
    results = []
    for k, (Xk, yk) in enumerate(train_nodes):
        idx = node_indices[k]
        if len(idx) == 0:
            results.append(None)
            continue
        Xk_node = Xk[idx]
        yk_node = yk[idx]
        left_mask = Xk_node[:, feature] <= threshold
        right_mask = ~left_mask
        lp = np.sum(yk_node[left_mask])
        lt = left_mask.sum()
        rp = np.sum(yk_node[right_mask])
        rt = right_mask.sum()
        pp = np.sum(yk_node)
        pt = len(yk_node)
        results.append((lp, lt, rp, rt, pp, pt))
    return results

def fiti_score(client_counts, lam):
    """Compute FITI score for a candidate split across all clients."""
    gains = []
    shifts = []
    for c in client_counts:
        if c is None: continue
        lp, lt, rp, rt, pp, pt = c
        if pt == 0: continue
        gain = compute_gini_gain(lp, lt, rp, rt, pp, pt)
        gains.append(gain)
        # Laplace-smoothed probability shift
        dp = (rp + 1) / (rt + 2) - (lp + 1) / (lt + 2)
        shifts.append(dp)
    if len(gains) == 0:
        return -np.inf
    return np.mean(gains) - lam * np.var(shifts)

def get_candidate_thresholds(train_nodes, feature, node_indices):
    """Get candidate thresholds from client percentiles."""
    all_vals = []
    for k, (Xk, yk) in enumerate(train_nodes):
        idx = node_indices[k]
        if len(idx) > 0:
            vals = Xk[idx, feature]
            all_vals.extend(vals.tolist())
    if len(all_vals) == 0:
        return []
    all_vals = np.array(sorted(set(all_vals)))
    # Use midpoints between sorted unique values, or percentiles
    if len(all_vals) <= 20:
        thresholds = all_vals[:-1]
    else:
        percs = np.percentile(all_vals, [10,20,30,40,50,60,70,80,90])
        thresholds = np.unique(percs)
    return thresholds

def build_tree_track_features(train_nodes, max_depth, lam, n_features,
                              inv_set, spu_set, noise_set):
    """
    Build FITI/FLSE tree and track feature selection at each depth.
    Returns list of (depth, feature_idx, feature_type) for each split node.
    """
    selected = []  # (depth, feat_idx, feat_type)

    def classify_feat(f):
        if f in spu_set: return "spurious"
        if f in inv_set: return "invariant"
        return "noise"

    def build(node_indices, depth):
        # Check stopping criteria
        total = sum(len(idx) for idx in node_indices)
        if total < 2 or depth >= max_depth:
            return
        all_labels = np.concatenate([train_nodes[k][1][node_indices[k]]
                                     for k in range(len(train_nodes))
                                     if len(node_indices[k]) > 0])
        if len(np.unique(all_labels)) == 1:
            return

        best_score = -np.inf
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            thresholds = get_candidate_thresholds(train_nodes, feat, node_indices)
            for thresh in thresholds:
                counts = get_client_counts(train_nodes, feat, thresh, node_indices)
                score = fiti_score(counts, lam)
                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            return

        ftype = classify_feat(best_feat)
        selected.append((depth, best_feat, ftype))

        # Recurse
        left_indices = []
        right_indices = []
        for k, (Xk, yk) in enumerate(train_nodes):
            idx = node_indices[k]
            if len(idx) == 0:
                left_indices.append(np.array([], dtype=int))
                right_indices.append(np.array([], dtype=int))
            else:
                lmask = Xk[idx, best_feat] <= best_thresh
                left_indices.append(idx[lmask])
                right_indices.append(idx[~lmask])
        build(left_indices, depth + 1)
        build(right_indices, depth + 1)

    init_indices = [np.arange(len(Xk)) for Xk, yk in train_nodes]
    build(init_indices, 0)
    return selected

def compute_freq(selections_list, feature_types):
    """Compute root and top-3 depth frequency for each feature type."""
    root_counts = {t: 0 for t in feature_types}
    top3_counts = {t: 0 for t in feature_types}
    n_seeds = len(selections_list)

    for selections in selections_list:
        root_sel = [s for s in selections if s[0] == 0]
        top3_sel = [s for s in selections if s[0] <= 2]
        for s in root_sel:
            ftype = s[2]
            root_counts[ftype] = root_counts.get(ftype, 0) + 1
        for s in top3_sel:
            ftype = s[2]
            top3_counts[ftype] = top3_counts.get(ftype, 0) + 1

    # Normalize by n_seeds for root
    n_top3_total = {t: 0 for t in feature_types}
    for selections in selections_list:
        top3_sel = [s for s in selections if s[0] <= 2]
        n = len(top3_sel)
        for t in feature_types:
            ct = sum(1 for s in top3_sel if s[2] == t)
            n_top3_total[t] = n_top3_total.get(t, 0) + (ct / n if n > 0 else 0)

    root_freq = {t: root_counts[t] / n_seeds for t in feature_types}
    top3_freq = {t: n_top3_total[t] / n_seeds for t in feature_types}
    return root_freq, top3_freq

seeds = list(range(10))
max_depth = 3
feature_types = ["invariant", "spurious", "noise"]

print("% === Table: Feature-selection frequency at root and top-3 levels ===")

# ---- Synthetic ----
print("\n% --- Synthetic Dataset ---")
syn_flse_sels = []
syn_fiti_sels = []
for seed in seeds:
    train_nodes, _, _, _, _, _ = get_synthetic_data(seed, mode="expanded")
    n_features = train_nodes[0][0].shape[1]
    inv_set = SYNTHETIC_INV
    spu_set = SYNTHETIC_SPU
    noise_set = SYNTHETIC_NOISE

    # FLSE (lambda=0)
    sels_flse = build_tree_track_features(train_nodes, max_depth, 0.0,
                                          n_features, inv_set, spu_set, noise_set)
    syn_flse_sels.append(sels_flse)

    # FITI (lambda=5 ~ typical for Synthetic)
    sels_fiti = build_tree_track_features(train_nodes, max_depth, 5.0,
                                          n_features, inv_set, spu_set, noise_set)
    syn_fiti_sels.append(sels_fiti)

    print(f"  Seed {seed}: FLSE root={[s for s in sels_flse if s[0]==0]}, FITI root={[s for s in sels_fiti if s[0]==0]}")

flse_root, flse_top3 = compute_freq(syn_flse_sels, feature_types)
fiti_root, fiti_top3 = compute_freq(syn_fiti_sels, feature_types)

print("\n% Synthetic: FLSE root freq:", flse_root, "top3:", flse_top3)
print("% Synthetic: FITI root freq:", fiti_root, "top3:", fiti_top3)

# Print as LaTeX table rows
print(f"\n% Synthetic & FLSE & {flse_root.get('invariant',0):.2f} & {flse_root.get('spurious',0):.2f} & {flse_root.get('noise',0):.2f} & {flse_top3.get('invariant',0):.2f} & {flse_top3.get('spurious',0):.2f} & {flse_top3.get('noise',0):.2f} \\\\")
print(f"% Synthetic & FITI & {fiti_root.get('invariant',0):.2f} & {fiti_root.get('spurious',0):.2f} & {fiti_root.get('noise',0):.2f} & {fiti_top3.get('invariant',0):.2f} & {fiti_top3.get('spurious',0):.2f} & {fiti_top3.get('noise',0):.2f} \\\\")

# Save for use
results_dict = {
    "Synthetic": {
        "FLSE": {"root": flse_root, "top3": flse_top3},
        "FITI": {"root": fiti_root, "top3": fiti_top3}
    }
}

# ---- Adult Spurious ----
print("\n% --- Adult Spurious Dataset ---")
adult_flse_sels = []
adult_fiti_sels = []
for seed in seeds:
    train_nodes, _, _, _, _, _ = get_adult_spurious_data(seed)
    n_features = train_nodes[0][0].shape[1]
    inv_set, spu_set, noise_set = adult_spu_types(n_features)

    sels_flse = build_tree_track_features(train_nodes, max_depth, 0.0,
                                          n_features, inv_set, spu_set, noise_set)
    adult_flse_sels.append(sels_flse)

    sels_fiti = build_tree_track_features(train_nodes, max_depth, 0.5,
                                          n_features, inv_set, spu_set, noise_set)
    adult_fiti_sels.append(sels_fiti)
    print(f"  Seed {seed}: FLSE root={[s for s in sels_flse if s[0]==0]}, FITI root={[s for s in sels_fiti if s[0]==0]}")

flse_root_a, flse_top3_a = compute_freq(adult_flse_sels, feature_types)
fiti_root_a, fiti_top3_a = compute_freq(adult_fiti_sels, feature_types)

print("\n% Adult Spurious: FLSE root freq:", flse_root_a, "top3:", flse_top3_a)
print("% Adult Spurious: FITI root freq:", fiti_root_a, "top3:", fiti_top3_a)
print(f"\n% Adult Spurious & FLSE & {flse_root_a.get('invariant',0):.2f} & {flse_root_a.get('spurious',0):.2f} & {flse_root_a.get('noise',0):.2f} & {flse_top3_a.get('invariant',0):.2f} & {flse_top3_a.get('spurious',0):.2f} & {flse_top3_a.get('noise',0):.2f} \\\\")
print(f"% Adult Spurious & FITI & {fiti_root_a.get('invariant',0):.2f} & {fiti_root_a.get('spurious',0):.2f} & {fiti_root_a.get('noise',0):.2f} & {fiti_top3_a.get('invariant',0):.2f} & {fiti_top3_a.get('spurious',0):.2f} & {fiti_top3_a.get('noise',0):.2f} \\\\")

results_dict["Adult_Spurious"] = {
    "FLSE": {"root": flse_root_a, "top3": flse_top3_a},
    "FITI": {"root": fiti_root_a, "top3": fiti_top3_a}
}

os.makedirs("exp", exist_ok=True)
with open("exp/feature_selection_freq.json", "w") as f:
    json.dump(results_dict, f, indent=2)
print("\n% Results saved to exp/feature_selection_freq.json")
