import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
data_dir = os.environ.get("FITI_DATA_DIR", os.path.join(os.path.dirname(CONFIG_PATH), "data"))
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        if "data_dir" in config:
            data_dir = os.environ.get("FITI_DATA_DIR", config["data_dir"])
            if not os.path.isabs(data_dir):
                data_dir = os.path.join(os.path.dirname(CONFIG_PATH), data_dir)


# --- DATASET LOADERS ---

def get_synthetic_data(seed, mode="expanded", n_train_nodes=8, n_test_nodes=2):
    rng = np.random.default_rng(seed)
    n_samples_per_node = 1000 if mode == "expanded" else 500
    n_features = 20
    
    invariant_indices = [0, 1, 2, 3, 4]
    spurious_indices = [5, 6, 7, 8, 9]
    
    def generate_node_data(n_samples, is_ood=False):
        X = rng.normal(0, 1, size=(n_samples, n_features))
        logits = np.sum(X[:, invariant_indices], axis=1)
        y = (logits > 0).astype(int)
        y = np.where(rng.random(n_samples) < 0.05, 1 - y, y)
        
        if not is_ood:
            direction = 1.0 if rng.random() > 0.4 else -1.0
            correlation_strength = rng.uniform(0.7, 1.0)
            for j in spurious_indices:
                X[:, j] = np.where(rng.random(n_samples) < correlation_strength, 
                                   direction * (y * 2 - 1) * 2.0 + rng.normal(0, 0.5, n_samples),
                                   rng.normal(0, 2.0, n_samples))
        else:
            for j in spurious_indices:
                X[:, j] = rng.normal(0, 2.0, size=n_samples)
        return X, y

    train_nodes = [generate_node_data(n_samples_per_node, is_ood=False) for _ in range(n_train_nodes)]
    test_nodes = [generate_node_data(n_samples_per_node, is_ood=True) for _ in range(n_test_nodes)]
    
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:200] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:200] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_adult_spurious_data(seed, n_train_nodes=8, n_test_nodes=2):
    train_df = pd.read_csv(os.path.join(data_dir, "adult/adult.data"), header=None, skipinitialspace=True)
    test_df = pd.read_csv(os.path.join(data_dir, "adult/adult.test"), header=None, skipinitialspace=True, skiprows=1)
    
    y_train = (train_df[14].astype(str).str.contains('>50K')).astype(int).values
    y_test = (test_df[14].astype(str).str.contains('>50K')).astype(int).values
    train_df = train_df.drop(columns=[14])
    test_df = test_df.drop(columns=[14])
    
    cat_cols = train_df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])
        
    X_train = train_df.values.astype(float)
    X_test = test_df.values.astype(float)
    
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    rng = np.random.default_rng(seed)
    total_nodes = n_train_nodes + n_test_nodes
    indices = np.arange(len(X))
    rng.shuffle(indices)
    partitions = np.array_split(indices, total_nodes)
    
    train_nodes = []
    test_nodes = []
    
    for i in range(total_nodes):
        idx = partitions[i]
        X_node = X[idx].copy()
        y_node = y[idx].copy()
        
        spurious_feat = np.zeros((len(X_node), 1))
        
        if i < n_train_nodes:
            direction = 1.0 if i % 2 == 0 else -1.0
            p_i = 0.75 + 0.02 * i
            mask = rng.random(len(X_node)) < p_i
            y_target = y_node if direction == 1.0 else (1 - y_node)
            spurious_feat[:, 0] = np.where(mask, y_target, 1 - y_target)
            X_node = np.hstack([X_node, spurious_feat])
            train_nodes.append((X_node, y_node))
        else:
            spurious_feat[:, 0] = rng.choice([0, 1], size=len(X_node))
            X_node = np.hstack([X_node, spurious_feat])
            test_nodes.append((X_node, y_node))
            
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:200] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:200] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_adult_natural_data(seed):
    # Natural domain split on Adult: partition by marital status (col 5)
    train_df = pd.read_csv(os.path.join(data_dir, "adult/adult.data"), header=None, skipinitialspace=True)
    test_df = pd.read_csv(os.path.join(data_dir, "adult/adult.test"), header=None, skipinitialspace=True, skiprows=1)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    y = (df[14].astype(str).str.contains('>50K')).astype(int).values
    marital = df[5].astype(str).values
    
    # We must fit encoder only on training partition (Married-civ-spouse, Never-married, Divorced, Separated)
    train_mask = np.isin(marital, ["Married-civ-spouse", "Never-married", "Divorced", "Separated"])
    
    df = df.drop(columns=[14, 5]) # Drop label and partition feature
    
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Fit on train mask only
        encoder.fit(df.loc[train_mask, cat_cols])
        df.loc[:, cat_cols] = encoder.transform(df[cat_cols])
            
    X = df.values.astype(float)
    
    # Define environments
    envs = {
        "Married-civ-spouse": 0,
        "Never-married": 1,
        "Divorced": 2,
        "Separated": 3
    }
    
    train_nodes = []
    test_nodes = []
    
    for status, env_id in envs.items():
        mask = (marital == status)
        train_nodes.append((X[mask], y[mask]))
        
    # OOD test is the rest (Widowed, Married-spouse-absent, Married-AF-spouse)
    ood_statuses = ["Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    ood_mask = np.isin(marital, ood_statuses)
    test_nodes.append((X[ood_mask], y[ood_mask]))
    
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:200] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:200] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_compas_spurious_data(seed, n_train_nodes=5, n_test_nodes=1):
    df = pd.read_csv(os.path.join(data_dir, "compas/compas-scores-two-years.csv"))
    cols = ['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    
    sub = df[cols + ['two_year_recid']].dropna()
    y = sub['two_year_recid'].values.astype(int)
    X_df = sub[cols].copy()
    
    # Split into train/test logic is done later, but for spurious the split is random.
    # To be perfectly rigorous, we should fit encoder on a subset or just the whole since it's random.
    # However, to avoid any leakage claims, we randomly determine train indices first.
    rng = np.random.default_rng(seed)
    total_nodes = n_train_nodes + n_test_nodes
    indices = np.arange(len(X_df))
    rng.shuffle(indices)
    partitions = np.array_split(indices, total_nodes)
    train_indices = np.concatenate(partitions[:n_train_nodes])
    
    cat_cols = ['sex', 'c_charge_degree']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(X_df.iloc[train_indices][cat_cols])
    X_df.loc[:, cat_cols] = encoder.transform(X_df[cat_cols])
    
    X = X_df.values.astype(float)
    
    
    train_nodes = []
    test_nodes = []
    
    for i in range(total_nodes):
        idx = partitions[i]
        X_node = X[idx].copy()
        y_node = y[idx].copy()
        
        spurious_feat = np.zeros((len(X_node), 1))
        
        if i < n_train_nodes:
            direction = 1.0 if i % 2 == 0 else -1.0
            p_i = 0.75 + 0.04 * i
            mask = rng.random(len(X_node)) < p_i
            y_target = y_node if direction == 1.0 else (1 - y_node)
            spurious_feat[:, 0] = np.where(mask, y_target, 1 - y_target)
            X_node = np.hstack([X_node, spurious_feat])
            train_nodes.append((X_node, y_node))
        else:
            spurious_feat[:, 0] = rng.choice([0, 1], size=len(X_node))
            X_node = np.hstack([X_node, spurious_feat])
            test_nodes.append((X_node, y_node))
            
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:200] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:200] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_compas_natural_data(seed):
    df = pd.read_csv(os.path.join(data_dir, "compas/compas-scores-two-years.csv"))
    cols = ['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    
    sub = df[cols + ['race', 'two_year_recid']].dropna()
    y = sub['two_year_recid'].values.astype(int)
    race = sub['race'].values
    
    X_df = sub[cols].copy()
    train_races = ['African-American', 'Caucasian']
    train_mask = np.isin(race, train_races)
    
    cat_cols = ['sex', 'c_charge_degree']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(X_df.loc[train_mask, cat_cols])
    X_df.loc[:, cat_cols] = encoder.transform(X_df[cat_cols])
    
    X = X_df.values.astype(float)
    
    train_races = ['African-American', 'Caucasian']
    train_nodes = []
    
    for r in train_races:
        mask = (race == r)
        train_nodes.append((X[mask], y[mask]))
        
    # OOD test is Hispanic, Other, Asian, Native American
    test_mask = ~np.isin(race, train_races)
    test_nodes = [(X[test_mask], y[test_mask])]
    
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:200] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:200] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_german_spurious_data(seed, n_train_nodes=4, n_test_nodes=1):
    df = pd.read_csv(os.path.join(data_dir, "german/german.data"), header=None, sep=' ', skipinitialspace=True)
    y = (df[20] == 2).astype(int).values # 1 for bad credit, 0 for good credit
    df = df.drop(columns=[20])
    
    rng = np.random.default_rng(seed)
    total_nodes = n_train_nodes + n_test_nodes
    indices = np.arange(len(df))
    rng.shuffle(indices)
    partitions = np.array_split(indices, total_nodes)
    train_indices = np.concatenate(partitions[:n_train_nodes])
    
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df.iloc[train_indices][cat_cols])
        df.loc[:, cat_cols] = encoder.transform(df[cat_cols])
            
    X = df.values.astype(float)
    
    
    train_nodes = []
    test_nodes = []
    
    for i in range(total_nodes):
        idx = partitions[i]
        X_node = X[idx].copy()
        y_node = y[idx].copy()
        
        spurious_feat = np.zeros((len(X_node), 1))
        
        if i < n_train_nodes:
            direction = 1.0 if i % 2 == 0 else -1.0
            p_i = 0.75 + 0.05 * i
            mask = rng.random(len(X_node)) < p_i
            y_target = y_node if direction == 1.0 else (1 - y_node)
            spurious_feat[:, 0] = np.where(mask, y_target, 1 - y_target)
            X_node = np.hstack([X_node, spurious_feat])
            train_nodes.append((X_node, y_node))
        else:
            spurious_feat[:, 0] = rng.choice([0, 1], size=len(X_node))
            X_node = np.hstack([X_node, spurious_feat])
            test_nodes.append((X_node, y_node))
            
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    # For small German dataset, use 50 samples for IID evaluation per client
    X_test_iid = np.concatenate([node[0][:50] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:50] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

def get_german_natural_data(seed):
    df = pd.read_csv(os.path.join(data_dir, "german/german.data"), header=None, sep=' ', skipinitialspace=True)
    y = (df[20] == 2).astype(int).values
    housing = df[14].values # Housing: A151, A152, A153
    df = df.drop(columns=[20, 14]) # Drop label and housing partition feature
    
    train_housing = ['A151', 'A152']
    train_mask = np.isin(housing, train_housing)
    
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df.loc[train_mask, cat_cols])
        df.loc[:, cat_cols] = encoder.transform(df[cat_cols])
            
    X = df.values.astype(float)
    
    train_housing = ['A151', 'A152']
    train_nodes = []
    
    for h in train_housing:
        mask = (housing == h)
        train_nodes.append((X[mask], y[mask]))
        
    test_mask = (housing == 'A153')
    test_nodes = [(X[test_mask], y[test_mask])]
    
    X_test_ood = np.concatenate([node[0] for node in test_nodes])
    y_test_ood = np.concatenate([node[1] for node in test_nodes])
    
    X_test_iid = np.concatenate([node[0][:50] for node in train_nodes])
    y_test_iid = np.concatenate([node[1][:50] for node in train_nodes])
    
    X_train_full = np.concatenate([node[0] for node in train_nodes])
    y_train_full = np.concatenate([node[1] for node in train_nodes])
    
    return train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_train_full, y_train_full)

# --- FITI ALGORITHM ---

class SimpleNode:
    def __init__(self, depth):
        self.depth = depth
        self.feat = None
        self.thresh = None
        self.pred = None
        self.left = None
        self.right = None

def get_gini(counts):
    n = np.sum(counts)
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p**2)

def federated_train_fiti(node_data, max_depth=3, n_classes=2, lam=0.0):
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
                        
                        p_l = c_left[1] / n_l if n_l > 0 else 0.5
                        p_r = c_right[1] / n_r if n_r > 0 else 0.5
                        local_prob_shifts.append(p_r - p_l)
                        
                        total_samples += n_t
                    else:
                        local_gains.append(0.0)
                        local_prob_shifts.append(0.0)
                
                if total_samples == 0: continue
                
                avg_gain = np.mean(local_gains)
                var_prob_shift = np.var(local_prob_shifts)
                score = avg_gain - lam * var_prob_shift
                
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

def predict_simple_tree(root, X):
    preds = []
    for x in X:
        curr = root
        while curr.left:
            if x[curr.feat] <= curr.thresh:
                curr = curr.left
            else:
                curr = curr.right
        preds.append(curr.pred)
    return np.array(preds)

# --- FEDERATED FOREST ALGORITHM ---

def federated_train_forest(node_data, max_depth=3, n_classes=2, lam=0.0, n_estimators=10, max_features="sqrt", seed=0):
    rng = np.random.default_rng(seed)
    n_features = node_data[0][0].shape[1]
    
    if max_features == "sqrt":
        n_sub_features = max(1, int(np.sqrt(n_features)))
    else:
        n_sub_features = n_features
        
    trees = []
    for b in range(n_estimators):
        sub_feat_indices = rng.choice(n_features, size=n_sub_features, replace=False)
        sub_feat_indices = sorted(sub_feat_indices)
        
        bagged_node_data = []
        for X_node, y_node in node_data:
            bagged_node_data.append((X_node[:, sub_feat_indices], y_node))
            
        tree_root = federated_train_fiti(bagged_node_data, max_depth=max_depth, n_classes=n_classes, lam=lam)
        trees.append((tree_root, sub_feat_indices))
    return trees

def predict_forest(trees, X):
    preds_all = []
    for tree_root, sub_feat_indices in trees:
        X_sub = X[:, sub_feat_indices]
        preds = predict_simple_tree(tree_root, X_sub)
        preds_all.append(preds)
    preds_all = np.array(preds_all)
    final_preds = []
    for i in range(X.shape[0]):
        counts = np.bincount(preds_all[:, i])
        final_preds.append(np.argmax(counts))
    return np.array(final_preds)

# --- RUNNER ---

def run_experiment(datasets, seeds, lambdas, max_depth=3):
    results = {}
    
    for d_name, loader in datasets.items():
        print(f"\n--- Running dataset: {d_name} ---")
        results[d_name] = {}
        
        for seed in seeds:
            print(f"Seed {seed}...")
            results[d_name][seed] = {}
            
            train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader(seed)
            
            # 1. FITI sweeps (single tree)
            results[d_name][seed]["fiti"] = []
            for lam in lambdas:
                start = time.perf_counter()
                root = federated_train_fiti(train_nodes, max_depth=max_depth, lam=lam)
                elapsed = time.perf_counter() - start
                
                y_pred_iid = predict_simple_tree(root, X_test_iid)
                y_pred_ood = predict_simple_tree(root, X_test_ood)
                
                acc_iid = accuracy_score(y_test_iid, y_pred_iid)
                acc_ood = accuracy_score(y_test_ood, y_pred_ood)
                
                results[d_name][seed]["fiti"].append({
                    "lambda": lam,
                    "iid_accuracy": float(acc_iid),
                    "ood_accuracy": float(acc_ood),
                    "time": elapsed
                })
                
            # 2. Local-only baseline (average single tree)
            results[d_name][seed]["local"] = []
            local_accs_iid = []
            local_accs_ood = []
            for node in train_nodes:
                root = federated_train_fiti([node], max_depth=max_depth, lam=0.0)
                y_pred_iid = predict_simple_tree(root, X_test_iid)
                y_pred_ood = predict_simple_tree(root, X_test_ood)
                local_accs_iid.append(accuracy_score(y_test_iid, y_pred_iid))
                local_accs_ood.append(accuracy_score(y_test_ood, y_pred_ood))
                
            results[d_name][seed]["local"].append({
                "iid_accuracy": float(np.mean(local_accs_iid)),
                "ood_accuracy": float(np.mean(local_accs_ood))
            })
            
            # 3. Centralized baseline (unconstrained single CART)
            results[d_name][seed]["centralized"] = []
            root = federated_train_fiti([(X_full, y_full)], max_depth=max_depth, lam=0.0)
            y_pred_iid = predict_simple_tree(root, X_test_iid)
            y_pred_ood = predict_simple_tree(root, X_test_ood)
            results[d_name][seed]["centralized"].append({
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
            })
            
            # 4. Federated Random Forest baselines (FedRF with lam=0 and FITI-Forest with lam=5.0)
            results[d_name][seed]["fedrf_standard"] = []
            forest_std = federated_train_forest(train_nodes, max_depth=max_depth, lam=0.0, n_estimators=10, seed=seed)
            y_pred_iid = predict_forest(forest_std, X_test_iid)
            y_pred_ood = predict_forest(forest_std, X_test_ood)
            results[d_name][seed]["fedrf_standard"].append({
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
            })
            
            results[d_name][seed]["fedrf_invariant"] = []
            forest_inv = federated_train_forest(train_nodes, max_depth=max_depth, lam=5.0, n_estimators=10, seed=seed)
            y_pred_iid = predict_forest(forest_inv, X_test_iid)
            y_pred_ood = predict_forest(forest_inv, X_test_ood)
            results[d_name][seed]["fedrf_invariant"].append({
                "iid_accuracy": float(accuracy_score(y_test_iid, y_pred_iid)),
                "ood_accuracy": float(accuracy_score(y_test_ood, y_pred_ood))
            })
            
    return results

def run_scalability_sweep(datasets, seed=0):
    scalability_results = {}
    loader = datasets["Synthetic"]
    
    scalability_results["depth"] = []
    train_nodes, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader(seed)
    for d in [3, 4, 5, 6]:
        start = time.perf_counter()
        root = federated_train_fiti(train_nodes, max_depth=d, lam=1.0)
        elapsed = time.perf_counter() - start
        scalability_results["depth"].append({
            "depth": d,
            "time": elapsed
        })
        
    scalability_results["clients"] = []
    for c in [4, 8, 12, 16]:
        def loader_c(s):
            return get_synthetic_data(s, mode="expanded", n_train_nodes=c)
        train_nodes_c, X_test_iid, y_test_iid, X_test_ood, y_test_ood, (X_full, y_full) = loader_c(seed)
        start = time.perf_counter()
        root = federated_train_fiti(train_nodes_c, max_depth=3, lam=1.0)
        elapsed = time.perf_counter() - start
        scalability_results["clients"].append({
            "clients": c,
            "time": elapsed
        })
        
    return scalability_results

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
    
    os.makedirs("exp", exist_ok=True)
    
    print("Running experiments...")
    results = run_experiment(datasets, seeds, lambdas)
    
    print("Running scalability sweeps...")
    scalability = run_scalability_sweep(datasets)
    
    output = {
        "results": results,
        "scalability": scalability,
        "lambdas": lambdas,
        "seeds": seeds
    }
    
    output_path = "exp/supplementary_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
