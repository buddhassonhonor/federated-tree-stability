import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def load_results():
    with open("exp/supplementary_results.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_priority_results():
    path = "exp/priority2_results.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_pareto_curves(results, lambdas, seeds):
    os.makedirs("figures", exist_ok=True)
    
    for d_name in results.keys():
        plt.figure(figsize=(10, 6))
        
        # FITI curve from fiti_sweeps
        mean_iids = []
        mean_oods = []
        for i, lam in enumerate(lambdas):
            iids = [results[d_name][str(s)]["fiti_sweeps"][i]["iid_accuracy"] for s in seeds]
            oods = [results[d_name][str(s)]["fiti_sweeps"][i]["ood_accuracy"] for s in seeds]
            mean_iids.append(np.mean(iids))
            mean_oods.append(np.mean(oods))
            
        plt.plot(lambdas, mean_oods, marker="o", label="FITI OOD Accuracy (Ours)", color="#1f77b4", linewidth=2.5, markersize=8)
        plt.plot(lambdas, mean_iids, marker="s", label="FITI IID Accuracy", color="#ff7f0e", linewidth=2, markersize=8, linestyle="--")
        
        # Local baseline
        local_oods = [results[d_name][str(s)]["local"][0]["ood_accuracy"] for s in seeds]
        plt.axhline(y=np.mean(local_oods), color="#d62728", linestyle=":", label="Local-only OOD", linewidth=2)
        
        # Centralized baseline
        cent_oods = [results[d_name][str(s)]["tuned_cart"][0]["ood_accuracy"] for s in seeds]
        plt.axhline(y=np.mean(cent_oods), color="#2ca02c", linestyle="-.", label="Centralized CART OOD", linewidth=2)
        
        plt.title(f"IID & OOD Accuracy vs. Lambda ($\\lambda$) on {d_name.replace('_', ' ')}")
        plt.xlabel("Invariance Penalty Strength ($\\lambda$)")
        plt.ylabel("Accuracy")
        plt.xscale("symlog", linthresh=0.1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"figures/pareto_{d_name.lower()}.png", dpi=100)
        plt.close()

def plot_grouped_accuracy_bars(results, priority_results, seeds):
    groups = [
        (
            "figures/accuracy_spurious_bars.png",
            ["Synthetic", "Adult_Spurious", "COMPAS_Spurious", "German_Spurious"],
            ["Syn.", "Adult-S", "COMPAS-S", "German-S"],
        ),
        (
            "figures/accuracy_natural_bars.png",
            ["Adult_Natural", "COMPAS_Natural", "German_Natural"],
            ["Adult-N", "COMPAS-N", "German-N"],
        ),
    ]
    model_specs = [
        ("flse", "FLSE", "#4e79a7"),
        ("fiti", "FITI", "#f28e2b"),
        ("tuned_cart", "CART-Pooled", "#59a14f"),
        ("client_vote", "Client-Vote", "#e15759"),
    ]

    for output_path, datasets, labels in groups:
        x = np.arange(len(datasets))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for offset, (key, label, color) in enumerate(model_specs):
            means = []
            stds = []
            for ds in datasets:
                if key == "client_vote":
                    vals = priority_results.get(ds, {}).get("client_ensemble_voting", {}).get("vals", [])
                    vals = np.array(vals, dtype=float)
                else:
                    vals = np.array([results[ds][str(s)][key][0]["ood_accuracy"] for s in seeds], dtype=float)
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            positions = x + (offset - 1.5) * width
            ax.bar(positions, means, width, yerr=stds, capsize=4, label=label, color=color, edgecolor="black", linewidth=0.4)

        ax.set_ylabel("OOD Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        ymin = max(0.0, min(ax.get_ylim()[0], 0.45))
        ax.set_ylim(ymin, min(1.0, ax.get_ylim()[1] + 0.03))
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()


def plot_lambda_sensitivity_split(results, lambdas, seeds):
    groups = [
        ("figures/lambda_sensitivity_spurious.png", ["Synthetic", "Adult_Spurious", "COMPAS_Spurious", "German_Spurious"], "Spurious-correlation settings"),
        ("figures/lambda_sensitivity_natural.png", ["Adult_Natural", "COMPAS_Natural", "German_Natural"], "Natural partitions"),
    ]
    label_map = {
        "Synthetic": "Syn.",
        "Adult_Spurious": "Adult-S",
        "COMPAS_Spurious": "COMPAS-S",
        "German_Spurious": "German-S",
        "Adult_Natural": "Adult-N",
        "COMPAS_Natural": "COMPAS-N",
        "German_Natural": "German-N",
    }
    xs = np.arange(len(lambdas))

    for output_path, datasets, title in groups:
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.4), sharey=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, ds in zip(axes, datasets):
            lam_to_accs = {lam: [] for lam in lambdas}
            selected_lambdas = []
            for s in seeds:
                for entry in results[ds][str(s)]["fiti_sweeps"]:
                    lam_to_accs[entry["lambda"]].append(entry["ood_accuracy"])
                selected_lambdas.append(results[ds][str(s)]["fiti"][0]["lambda"])
            means = [np.mean(lam_to_accs[lam]) for lam in lambdas]
            stds = [np.std(lam_to_accs[lam]) for lam in lambdas]
            ax.plot(xs, means, marker="o", color="#4e79a7", linewidth=2)
            ax.fill_between(xs, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color="#4e79a7", alpha=0.16)
            chosen = np.median(selected_lambdas)
            if chosen in lambdas:
                ax.axvline(lambdas.index(chosen), color="#d62728", linestyle="--", linewidth=1.4)
            ax.set_title(label_map.get(ds, ds))
            ax.set_xlabel("$\\lambda$")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(lam) for lam in lambdas], rotation=45)
            ax.set_ylabel("OOD Accuracy")
            ax.grid(True, linestyle="--", alpha=0.3)
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()

def plot_scalability(scalability):
    # Depth Scalability
    depth_data = scalability["depth"]
    depths = [d["depth"] for d in depth_data]
    depth_times = [d["time"] for d in depth_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, depth_times, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    plt.title("Training Time vs. Tree Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/scalability_depth.png", dpi=100)
    plt.close()
    
    # Client Scalability
    client_data = scalability["clients"]
    clients = [c["clients"] for c in client_data]
    client_times = [c["time"] for c in client_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(clients, client_times, marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
    plt.title("Training Time vs. Number of Client Nodes")
    plt.xlabel("Number of Clients")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/scalability_clients.png", dpi=100)
    plt.close()

def main():
    data = load_results()
    results = data["results"]
    scalability = data["scalability"]
    lambdas = data["lambdas"]
    seeds = data["seeds"]
    priority_results = load_priority_results()
    
    plot_pareto_curves(results, lambdas, seeds)
    plot_grouped_accuracy_bars(results, priority_results, seeds)
    plot_lambda_sensitivity_split(results, lambdas, seeds)
    plot_scalability(scalability)
    print("Plotting completed successfully.")

if __name__ == "__main__":
    main()
