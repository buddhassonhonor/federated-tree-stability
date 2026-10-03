import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy.stats import wilcoxon

# We import the loaders directly or rewrite them cleanly to avoid relative path issues.
# Since we know run_supplementary_experiments.py is in src/, let's import the data loaders.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.run_supplementary_experiments import (
    get_synthetic_data,
    get_adult_spurious_data,
    get_adult_natural_data,
    get_compas_spurious_data,
    get_compas_natural_data,
    get_german_spurious_data,
    get_german_natural_data,
    SimpleNode,
    get_gini,
    predict_simple_tree,
    federated_train_forest,
    predict_forest
)

# Robust Laplace-smoothed probability shift calculation
def federated_train_fiti_robust(node_data, max_depth=3, n_classes=2, lam=0.0, penalty_type="variance"):
    n_features = node_data[0][0].shape[1]
    
    def find_best_split(subset_indices_per_node):
        best_score = -1e9
        best_split = None
        
        for feat in range(n_features):
            all_vals = []
            for i, (X_node, _) in enumerate(node_data):
                if len(subset_indices_per_node[i]) > 0:
                    all_vals.extend(np.percentile(X_node[subset_indices_per_node[i], feat], [25, 50, 75]))
            
            thresholds = np.unique(all_vals)
            if len(thresholds) == 0: continue
            
            for thresh in thresholds:
                local_gains = []
                local_prob_shifts = []
                total_samples = 0
                
                for i, (X_node, y_node) in enumerate(node_data):
                    idx = subset_indices_per_node[i]
                    if len(idx) == 0:
                        local_gains.append(0.0)
                        local_prob_shifts.append(0.0)
                        continue
                    
                    mask = X_node[idx, feat] <= thresh
                    y_sub = y_node[idx]
                    
                    n_l = np.sum(mask)
                    n_r = len(mask) - n_l
                    n_t = len(mask)
                    
                    if n_l > 0 and n_r > 0:
                        c_left = np.bincount(y_sub[mask], minlength=n_classes)
                        c_right = np.bincount(y_sub[~mask], minlength=n_classes)
                        
                        g_parent = get_gini(np.bincount(y_sub, minlength=n_classes))
                        g_children = (n_l / n_t) * get_gini(c_left) + (n_r / n_t) * get_gini(c_right)
                        local_gains.append(g_parent - g_children)
                        
                        # Laplace smoothing: (positive + 1) / (total + 2) to handle edge cases
                        p_l = (c_left[1] + 1) / (n_l + 2)
                        p_r = (c_right[1] + 1) / (n_r + 2)
                        local_prob_shifts.append(p_r - p_l)
                        
                        total_samples += n_t
                    else:
                        local_gains.append(0.0)
                        local_prob_shifts.append(0.0)
                
                if total_samples == 0: continue
                
                avg_gain = np.mean(local_gains)
                
                if penalty_type == "variance":
                    penalty = np.var(local_prob_shifts)
                elif penalty_type == "abs_range":
                    penalty = np.max(local_prob_shifts) - np.min(local_prob_shifts)
                elif penalty_type == "mad":
                    # Mean Absolute Deviation
                    penalty = np.mean(np.abs(np.array(local_prob_shifts) - np.mean(local_prob_shifts)))
                else:
                    penalty = 0.0
                    
                score = avg_gain - lam * penalty
                
                if score > best_score:
                    best_score = score
                    best_split = (feat, thresh)
                    
        return best_split

    def build_tree(subset_indices_per_node, depth):
        node = SimpleNode(depth)
        
        global_counts = np.zeros(n_classes)
        for i, (_, y_node) in enumerate(node_data):
            if len(subset_indices_per_node[i]) > 0:
                global_counts += np.bincount(y_node[subset_indices_per_node[i]], minlength=n_classes)
        
        if np.sum(global_counts) == 0:
            node.pred = 0
            return node
            
        node.pred = np.argmax(global_counts)
        
        if depth >= max_depth or np.max(global_counts) == np.sum(global_counts):
            return node
        
        split = find_best_split(subset_indices_per_node)
        if split is None:
            return node
        
        feat, thresh = split
        node.feat = feat
        node.thresh = thresh
        
        left_indices = []
        right_indices = []
        for i, (X_node, _) in enumerate(node_data):
            idx = subset_indices_per_node[i]
            if len(idx) == 0:
                left_indices.append(np.array([], dtype=int))
                right_indices.append(np.array([], dtype=int))
            else:
                mask = X_node[idx, feat] <= thresh
                left_indices.append(idx[mask])
                right_indices.append(idx[~mask])
            
        node.left = build_tree(left_indices, depth + 1)
        node.right = build_tree(right_indices, depth + 1)
        return node

    initial_indices = [np.arange(len(X_node)) for X_node, _ in node_data]
    return build_tree(initial_indices, 0)

# Tuned Centralized Baselines
def train_tuned_cart(X, y, max_depths=[3, 5, 8, None], min_samples_splits=[2, 5, 10]):
    best_clf = None
    best_score = -1
    # Simple validation split internally
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    for d in max_depths:
        for s in min_samples_splits:
            clf = DecisionTreeClassifier(max_depth=d, min_samples_split=s, random_state=42)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score = score
                best_clf = (d, s)
                
    # Retrain on full
    clf = DecisionTreeClassifier(max_depth=best_clf[0], min_samples_split=best_clf[1], random_state=42)
    clf.fit(X, y)
    return clf, best_clf

def train_tuned_rf(X, y, max_depths=[3, 5, 8, None], n_estimators=[50, 100]):
    best_clf = None
    best_score = -1
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    for d in max_depths:
        for n in n_estimators:
            clf = RandomForestClassifier(max_depth=d, n_estimators=n, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score = score
                best_clf = (d, n)
                
    clf = RandomForestClassifier(max_depth=best_clf[0], n_estimators=best_clf[1], random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf, best_clf

def train_tuned_extratrees(X, y, max_depths=[3, 5, 8, None], n_estimators=[50, 100]):
    best_clf = None
    best_score = -1
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    for d in max_depths:
        for n in n_estimators:
            clf = ExtraTreesClassifier(max_depth=d, n_estimators=n, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score = score
                best_clf = (d, n)
                
    clf = ExtraTreesClassifier(max_depth=best_clf[0], n_estimators=best_clf[1], random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf, best_clf

def train_tuned_gbdt(X, y, learning_rates=[0.05, 0.1, 0.2], n_estimators=[50, 100]):
    best_clf = None
    best_score = -1
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    for lr in learning_rates:
        for n in n_estimators:
            clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score = score
                best_clf = (lr, n)
                
    clf = GradientBoostingClassifier(learning_rate=best_clf[0], n_estimators=best_clf[1], random_state=42)
    clf.fit(X, y)
    return clf, best_clf

def train_tuned_xgboost(X, y, learning_rates=[0.05, 0.1, 0.2], n_estimators=[50, 100]):
    best_clf = None
    best_score = -1
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]
    
    for lr in learning_rates:
        for n in n_estimators:
            clf = XGBClassifier(learning_rate=lr, n_estimators=n, random_state=42, n_jobs=-1, eval_metric="logloss")
            clf.fit(X_train, y_train)
            score = accuracy_score(y_val, clf.predict(X_val))
            if score > best_score:
                best_score = score
                best_clf = (lr, n)
                
    clf = XGBClassifier(learning_rate=best_clf[0], n_estimators=best_clf[1], random_state=42, n_jobs=-1, eval_metric="logloss")
    clf.fit(X, y)
    return clf, best_clf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    datasets = {
        "Synthetic": lambda s: get_synthetic_data(s, mode="expanded" if not args.dry_run else "quick"),
        "Adult_Spurious": lambda s: get_adult_spurious_data(s),
        "Adult_Natural": lambda s: get_adult_natural_data(s),
        "COMPAS_Spurious": lambda s: get_compas_spurious_data(s),
        "COMPAS_Natural": lambda s: get_compas_natural_data(s),
        "German_Spurious": lambda s: get_german_spurious_data(s),
        "German_Natural": lambda s: get_german_natural_data(s),
    }
    
    seeds = [0] if args.dry_run else list(range(10))
    lambdas = [0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    results = {}
    
    # Validation splits for training clients (to select lambda)
    # Synthetic (8 clients): train 0-5, val 6-7
    # Adult Spurious (8 clients): train 0-5, val 6-7
    # Adult Natural (4 clients): train 0-2, val 3
    # COMPAS Spurious (5 clients): train 0-3, val 4
    # COMPAS Natural (2 clients): train 0, val 1
    # German Spurious (4 clients): train 0-2, val 3
    # German Natural (2 clients): train 0, val 1
    val_splits = {
        "Synthetic": (6, 8),
        "Adult_Spurious": (6, 8),
        "Adult_Natural": (3, 4),
        "COMPAS_Spurious": (4, 5),
        "COMPAS_Natural": (1, 2),
        "German_Spurious": (3, 4),
        "German_Natural": (1, 2),
    }
    
    for d_name, loader in datasets.items():
        print(f"\n--- Running dataset: {d_name} ---")
        results[d_name] = {}
        
        split_idx, total_nodes = val_splits[d_name]
        
        for seed in seeds:
            print(f"Seed {seed}...")
            results[d_name][seed] = {}
            
            train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader(seed)
            
            # --- LAMBDA SELECTION ---
            # Train validation FITI models on train_nodes[:split_idx]
            val_train_nodes = train_nodes[:split_idx]
            val_test_nodes = train_nodes[split_idx:total_nodes]
            
            X_val_test = np.concatenate([node[0] for node in val_test_nodes])
            y_val_test = np.concatenate([node[1] for node in val_test_nodes])
            
            best_val_score = -1.0
            selected_lambda = 0.0
            
            for lam in lambdas:
                val_root = federated_train_fiti_robust(val_train_nodes, max_depth=3, lam=lam)
                val_preds = predict_simple_tree(val_root, X_val_test)
                val_acc = accuracy_score(y_val_test, val_preds)
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    selected_lambda = lam
            
            print(f"Selected Lambda for Seed {seed}: {selected_lambda} (Val Accuracy: {best_val_score:.4f})")
            
            # --- FINAL MODEL WITH SELECTED LAMBDA ---
            # Train on all train_nodes
            start = time.perf_counter()
            fiti_root = federated_train_fiti_robust(train_nodes, max_depth=3, lam=selected_lambda)
            elapsed = time.perf_counter() - start
            
            y_pred_iid = predict_simple_tree(fiti_root, X_test_iid)
            y_pred_ood = predict_simple_tree(fiti_root, X_test_ood)
            
            acc_iid = accuracy_score(y_test_iid, y_pred_iid)
            acc_ood = accuracy_score(y_test_ood, y_pred_ood)
            
            results[d_name][seed]["fiti"] = [{
                "lambda": selected_lambda,
                "iid_accuracy": float(acc_iid),
                "ood_accuracy": float(acc_ood),
                "time": elapsed
            }]
            
            # FITI lambda=0.0 control (FLSE)
            flse_root = federated_train_fiti_robust(train_nodes, max_depth=3, lam=0.0)
            y_pred_flse_iid = predict_simple_tree(flse_root, X_test_iid)
            y_pred_flse_ood = predict_simple_tree(flse_root, X_test_ood)
            results[d_name][seed]["flse"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_flse_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_flse_ood))
            }]
            
            # FITI sweeps (for pareto curve plots)
            results[d_name][seed]["fiti_sweeps"] = []
            for lam in lambdas:
                root = federated_train_fiti_robust(train_nodes, max_depth=3, lam=lam)
                y_pred_iid = predict_simple_tree(root, X_test_iid)
                y_pred_ood = predict_simple_tree(root, X_test_ood)
                results[d_name][seed]["fiti_sweeps"].append({
                    "lambda": lam,
                    "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                    "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
                })
                
            # Local-only baseline
            local_accs_iid = []
            local_accs_ood = []
            for node in train_nodes:
                root = federated_train_fiti_robust([node], max_depth=3, lam=0.0)
                y_pred_iid = predict_simple_tree(root, X_test_iid)
                y_pred_ood = predict_simple_tree(root, X_test_ood)
                local_accs_iid.append(accuracy_score(y_test_iid, y_pred_iid))
                local_accs_ood.append(accuracy_score(y_test_ood, y_pred_ood))
                
            results[d_name][seed]["local"] = [{
                "iid_accuracy": float(np.mean(local_accs_iid)),
                "ood_accuracy": float(np.mean(local_accs_ood))
            }]
            
            # --- TUNED CENTRALIZED BASELINES ---
            # 1. CART
            cart_clf, best_cart_params = train_tuned_cart(X_full, y_full)
            results[d_name][seed]["tuned_cart"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, cart_clf.predict(X_test_iid))),
                "ood_accuracy": float(accuracy_score(y_test_ood, cart_clf.predict(X_test_ood))),
                "params": best_cart_params
            }]
            
            # 2. Random Forest
            rf_clf, best_rf_params = train_tuned_rf(X_full, y_full)
            results[d_name][seed]["tuned_rf"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, rf_clf.predict(X_test_iid))),
                "ood_accuracy": float(accuracy_score(y_test_ood, rf_clf.predict(X_test_ood))),
                "params": best_rf_params
            }]
            
            # 3. ExtraTrees
            et_clf, best_et_params = train_tuned_extratrees(X_full, y_full)
            results[d_name][seed]["tuned_et"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, et_clf.predict(X_test_iid))),
                "ood_accuracy": float(accuracy_score(y_test_ood, et_clf.predict(X_test_ood))),
                "params": best_et_params
            }]
            
            # 4. GBDT
            gbdt_clf, best_gbdt_params = train_tuned_gbdt(X_full, y_full)
            results[d_name][seed]["tuned_gbdt"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, gbdt_clf.predict(X_test_iid))),
                "ood_accuracy": float(accuracy_score(y_test_ood, gbdt_clf.predict(X_test_ood))),
                "params": best_gbdt_params
            }]
            
            # 5. XGBoost
            xgb_clf, best_xgb_params = train_tuned_xgboost(X_full, y_full)
            results[d_name][seed]["tuned_xgb"] = [{
                "iid_accuracy": float(accuracy_score(y_test_iid, xgb_clf.predict(X_test_iid))),
                "ood_accuracy": float(accuracy_score(y_test_ood, xgb_clf.predict(X_test_ood))),
                "params": best_xgb_params
            }]
            
            # Ensembles (FedRF and FITI-Forest with selected lambda)
            results[d_name][seed]["fedrf_standard"] = []
            forest_std = federated_train_forest(train_nodes, max_depth=3, lam=0.0, n_estimators=10, seed=seed)
            y_pred_iid = predict_forest(forest_std, X_test_iid)
            y_pred_ood = predict_forest(forest_std, X_test_ood)
            results[d_name][seed]["fedrf_standard"].append({
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
            })
            
            results[d_name][seed]["fedrf_invariant"] = []
            # FITI-Forest using selected lambda
            forest_inv = federated_train_forest(train_nodes, max_depth=3, lam=selected_lambda, n_estimators=10, seed=seed)
            y_pred_iid = predict_forest(forest_inv, X_test_iid)
            y_pred_ood = predict_forest(forest_inv, X_test_ood)
            results[d_name][seed]["fedrf_invariant"].append({
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
            })
            
            # --- FITI ABLATIONS ---
            # 1. Shuffled client labels control (variance penalty calculated on shuffled y_node)
            shuffled_train_nodes = []
            for X_node, y_node in train_nodes:
                y_shuffled = y_node.copy()
                np.random.default_rng(seed).shuffle(y_shuffled)
                shuffled_train_nodes.append((X_node, y_shuffled))
            
            ab_root_shuf = federated_train_fiti_robust(shuffled_train_nodes, max_depth=3, lam=selected_lambda)
            results[d_name][seed]["ab_shuffled"] = [{
                "ood_accuracy": float(accuracy_score(y_test_ood, predict_simple_tree(ab_root_shuf, X_test_ood)))
            }]
            
            # 2. Random Client Grouping control
            # Concatenate training data and split randomly into the same sizes as original environments
            X_all = np.concatenate([node[0] for node in train_nodes])
            y_all = np.concatenate([node[1] for node in train_nodes])
            indices = np.arange(len(X_all))
            np.random.default_rng(seed).shuffle(indices)
            
            rand_train_nodes = []
            start_i = 0
            for node in train_nodes:
                end_i = start_i + len(node[0])
                idx = indices[start_i:end_i]
                rand_train_nodes.append((X_all[idx], y_all[idx]))
                start_i = end_i
                
            ab_root_rand = federated_train_fiti_robust(rand_train_nodes, max_depth=3, lam=selected_lambda)
            results[d_name][seed]["ab_rand_group"] = [{
                "ood_accuracy": float(accuracy_score(y_test_ood, predict_simple_tree(ab_root_rand, X_test_ood)))
            }]
            
            # 3. FITI with Absolute Shift Range penalty
            ab_root_range = federated_train_fiti_robust(train_nodes, max_depth=3, lam=selected_lambda, penalty_type="abs_range")
            results[d_name][seed]["ab_abs_range"] = [{
                "ood_accuracy": float(accuracy_score(y_test_ood, predict_simple_tree(ab_root_range, X_test_ood)))
            }]
            
            # 4. FITI with Mean Absolute Deviation (MAD) penalty
            ab_root_mad = federated_train_fiti_robust(train_nodes, max_depth=3, lam=selected_lambda, penalty_type="mad")
            results[d_name][seed]["ab_mad"] = [{
                "ood_accuracy": float(accuracy_score(y_test_ood, predict_simple_tree(ab_root_mad, X_test_ood)))
            }]
            
    # Scalability Sweeps (on Synthetic)
    print("Running scalability sweeps...")
    scalability_results = {}
    loader = datasets["Synthetic"]
    
    scalability_results["depth"] = []
    train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader(0)
    for d in [3, 4, 5, 6]:
        start = time.perf_counter()
        root = federated_train_fiti_robust(train_nodes, max_depth=d, lam=1.0)
        elapsed = time.perf_counter() - start
        scalability_results["depth"].append({
            "depth": d,
            "time": elapsed
        })
        
    scalability_results["clients"] = []
    for c in [4, 8, 12, 16]:
        def loader_c(s):
            return get_synthetic_data(s, mode="expanded", n_train_nodes=c)
        train_nodes_c, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader_c(0)
        start = time.perf_counter()
        root = federated_train_fiti_robust(train_nodes_c, max_depth=3, lam=1.0)
        elapsed = time.perf_counter() - start
        scalability_results["clients"].append({
            "clients": c,
            "time": elapsed
        })
        
    output = {
        "results": results,
        "scalability": scalability_results,
        "lambdas": lambdas,
        "seeds": seeds
    }
    
    os.makedirs("exp", exist_ok=True)
    output_path = "exp/supplementary_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"Benchmark experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
