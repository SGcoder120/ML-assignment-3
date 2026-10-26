# Run the experiment grid, test the best configs, plot learning curves and run the baseline
# Usage: python run_experiments.py --config experiments.json --out results
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from nn import NeuralNet, load_data

# Same values as experiments.json, only used if that file is missing
DEFAULT_CONFIG = {
    "seed": 42, "split": [0.70, 0.15, 0.15], "epochs": 100, "batch_size": 64,
    "patience": 15, "l2": 0.0,
    "activations": ["sigmoid", "tanh", "relu"],
    "architectures": [[64], [64, 32]],
    "optimizers": {"sgd": {"lrs": [0.1, 0.01]}, "adam": {"lrs": [0.01, 0.001]}},
    "selection_metric": "val_macro_f1",
}

SUMMARY_COLUMNS = ["train_acc", "val_acc", "val_macro_f1", "best_epoch"]


def plot_curve(ax, net, row):
    # Training and validation loss for one trained network
    epochs = np.arange(1, len(net.history["train_loss"]) + 1)
    ax.plot(epochs, net.history["train_loss"], label="train loss")
    ax.plot(epochs, net.history["val_loss"], label="validation loss")
    ax.axvline(net.best_epoch, color="gray", ls=":", label=f"best epoch ({net.best_epoch})")
    ax.set_title(f"{row.activation} | hidden {row.hidden} | {row.optimizer}, lr={row.lr}", fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments.json")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        print(f"{args.config} not found, using the built-in default config")
        cfg = DEFAULT_CONFIG
    os.makedirs(args.out, exist_ok=True)

    # Preprocess once so every run uses the same split
    data = NeuralNet(seed=cfg["seed"]).preprocess(
        load_data(), split=tuple(cfg["split"]), seed=cfg["seed"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"features after preprocessing: {X_train.shape[1]} | sizes "
          f"{len(y_train)}/{len(y_val)}/{len(data['y_test'])} | classes {data['classes']}")

    # Train every combination of activation, architecture, optimizer and learning rate
    grid = [(act, arch, opt, lr)
            for act in cfg["activations"]
            for arch in cfg["architectures"]
            for opt, settings in cfg["optimizers"].items()
            for lr in settings["lrs"]]
    nets, rows = [], []
    for i, (act, arch, opt, lr) in enumerate(grid, 1):
        net = NeuralNet(hidden=arch, activation=act, optimizer=opt, lr=lr,
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                        patience=cfg["patience"], l2=cfg["l2"], seed=cfg["seed"])
        net.train(X_train, y_train, X_val, y_val)
        train, val = net.evaluate(X_train, y_train), net.evaluate(X_val, y_val)
        rows.append({
            "id": i, "activation": act, "hidden": "-".join(map(str, arch)),
            "optimizer": opt, "lr": lr, "epochs_run": net.epochs_run,
            "best_epoch": net.best_epoch,
            "train_acc": train["accuracy"], "val_acc": val["accuracy"],
            "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"],
            "train_loss": train["loss"], "val_loss": val["loss"],
        })
        nets.append(net)
        print(f"[{i:2d}/{len(grid)}] {act:7s} {str(arch):9s} {opt:5s} lr={lr:<6} "
              f"train {train['accuracy']:.4f} val {val['accuracy']:.4f} "
              f"valF1 {val['macro_f1']:.4f} best_ep {net.best_epoch:3d}")
    table = pd.DataFrame(rows)
    table.to_csv(f"{args.out}/results_table.csv", index=False)

    # Pick the best config for each activation using validation data only
    best = {}
    for act in cfg["activations"]:
        candidates = [row for row in rows if row["activation"] == act]
        best[act] = max(candidates, key=lambda row: row[cfg["selection_metric"]])["id"]

    # The test set is only used here, for those best configs
    test_rows = []
    for act, idx in best.items():
        row = table.iloc[idx - 1]
        test = nets[idx - 1].evaluate(data["X_test"], data["y_test"])
        test_rows.append({"activation": act, "config_id": idx, "hidden": row.hidden,
                          "optimizer": row.optimizer, "lr": row.lr,
                          "val_acc": row.val_acc, "val_macro_f1": row.val_macro_f1,
                          "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"]})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(f"{args.out}/best_per_activation.csv", index=False)

    # Learning curves: all three together, then one file per activation
    fig, axes = plt.subplots(1, len(best), figsize=(5 * len(best), 3.8), sharey=True)
    for ax, idx in zip(np.atleast_1d(axes), best.values()):
        plot_curve(ax, nets[idx - 1], table.iloc[idx - 1])
    fig.tight_layout()
    fig.savefig(f"{args.out}/learning_curves.png", dpi=170)
    plt.close(fig)
    for act, idx in best.items():
        fig, ax = plt.subplots(figsize=(5, 3.6))
        plot_curve(ax, nets[idx - 1], table.iloc[idx - 1])
        fig.tight_layout()
        fig.savefig(f"{args.out}/learning_curve_{act}.png", dpi=170)
        plt.close(fig)

    # Logistic regression baseline on the same split
    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    pred = model.predict(data["X_test"])
    baseline = {"test_acc": float(np.mean(pred == data["y_test"])),
                "test_macro_f1": float(f1_score(data["y_test"], pred, average="macro")),
                "val_acc": float(model.score(X_val, y_val))}
    summary = {"seed": cfg["seed"], "n_features": int(X_train.shape[1]),
               "baseline": baseline, "best_ids": best}
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Average results by activation, optimizer and architecture (used in the report)
    with open(f"{args.out}/summaries.txt", "w") as f:
        for name, column in [("by_activation", "activation"), ("by_optimizer", "optimizer"),
                             ("by_arch", "hidden")]:
            means = pd.DataFrame(table.groupby(column)[SUMMARY_COLUMNS].mean())
            f.write(f"== {name} ==\n{means.round(4).to_string()}\n\n")

    # Markdown copy of the tables
    with open(f"{args.out}/results_table.md", "w") as f:
        f.write(table.round(4).to_markdown(index=False) + "\n\n")
        f.write(test_table.round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Logistic regression baseline: {json.dumps(baseline)}\n")

    print("\nBest per activation (test set):\n", test_table.round(4).to_string(index=False))
    print("\nBaseline:", baseline)


if __name__ == "__main__":
    main()