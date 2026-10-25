"""Run the full experiment grid, pick the best config per activation (by validation),
evaluate those on the test set, draw learning curves and run the logistic-regression baseline.

    python run_experiments.py --config experiments.json --csv data.csv --out results
"""
import argparse
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from nn import NeuralNet, load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments.json")
    p.add_argument("--csv", default=None, help="optional CSV path (default: ucimlrepo download)")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)

    prep = NeuralNet(seed=cfg["seed"])
    data = prep.preprocess(load_data(a.csv), split=tuple(cfg["split"]), seed=cfg["seed"])
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    print(f"features after preprocessing: {Xtr.shape[1]} | sizes "
          f"{len(ytr)}/{len(yva)}/{len(data['y_test'])} | classes {data['classes']}")

    # ---------------- grid of experiments ----------------
    runs, rows = [], []
    grid = [(act, arch, opt, lr)
            for act in cfg["activations"] for arch in cfg["architectures"]
            for opt, o in cfg["optimizers"].items() for lr in o["lrs"]]
    for i, (act, arch, opt, lr) in enumerate(grid, 1):
        t0 = time.time()
        net = NeuralNet(hidden=arch, activation=act, optimizer=opt, lr=lr,
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                        patience=cfg["patience"], l2=cfg["l2"], seed=cfg["seed"])
        net.train(Xtr, ytr, Xva, yva)
        tr, va = net.evaluate(Xtr, ytr), net.evaluate(Xva, yva)
        rows.append({
            "id": i, "activation": act, "hidden": "-".join(map(str, arch)),
            "optimizer": opt, "lr": lr, "epochs_run": net.epochs_run,
            "best_epoch": net.best_epoch,
            "train_acc": tr["accuracy"], "val_acc": va["accuracy"],
            "train_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"],
            "train_loss": tr["loss"], "val_loss": va["loss"],
        })
        runs.append(net)
        print(f"[{i:2d}/{len(grid)}] {act:7s} {str(arch):9s} {opt:5s} lr={lr:<6} "
              f"train {tr['accuracy']:.4f} val {va['accuracy']:.4f} "
              f"valF1 {va['macro_f1']:.4f} best_ep {net.best_epoch:3d} ({time.time()-t0:.1f}s)")

    table = pd.DataFrame(rows)
    table.to_csv(f"{a.out}/results_table.csv", index=False)

    # ---------------- best per activation (validation only) ----------------
    metric = cfg["selection_metric"]
    best = {}
    for act in cfg["activations"]:
        sub = table[table.activation == act]
        best[act] = int(sub.loc[sub[metric].idxmax(), "id"])

    test_rows = []
    for act, idx in best.items():
        net, row = runs[idx - 1], table.iloc[idx - 1]
        te = net.evaluate(data["X_test"], data["y_test"])     # test set used only here
        test_rows.append({"activation": act, "config_id": idx, "hidden": row.hidden,
                          "optimizer": row.optimizer, "lr": row.lr,
                          "val_acc": row.val_acc, "val_macro_f1": row.val_macro_f1,
                          "test_acc": te["accuracy"], "test_macro_f1": te["macro_f1"]})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(f"{a.out}/best_per_activation.csv", index=False)

    # ---------------- learning curves ----------------
    fig, axes = plt.subplots(1, len(best), figsize=(5 * len(best), 3.8), sharey=True)
    for ax, (act, idx) in zip(np.atleast_1d(axes), best.items()):
        net, row = runs[idx - 1], table.iloc[idx - 1]
        h = net.history
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], label="train loss")
        ax.plot(ep, h["val_loss"], label="validation loss")
        ax.axvline(net.best_epoch, color="gray", ls=":", label=f"best epoch ({net.best_epoch})")
        ax.set_title(f"{act} | hidden {row.hidden} | {row.optimizer}, lr={row.lr}", fontsize=9)
        ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("cross-entropy loss")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out}/learning_curves.png", dpi=170)
    plt.close()
    for act, idx in best.items():          # one figure per activation as well
        h = runs[idx - 1].history
        plt.figure(figsize=(5, 3.6))
        ep = np.arange(1, len(h["train_loss"]) + 1)
        plt.plot(ep, h["train_loss"], label="train"); plt.plot(ep, h["val_loss"], label="validation")
        plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.title(f"{act} (best config)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{a.out}/learning_curve_{act}.png", dpi=170); plt.close()

    # ---------------- logistic regression baseline (same split) ----------------
    lr_model = LogisticRegression(max_iter=3000)
    lr_model.fit(Xtr, ytr)
    pred = lr_model.predict(data["X_test"])
    baseline = {"test_acc": float(np.mean(pred == data["y_test"])),
                "test_macro_f1": float(f1_score(data["y_test"], pred, average="macro")),
                "val_acc": float(lr_model.score(Xva, yva))}
    json.dump({"seed": cfg["seed"], "n_features": int(Xtr.shape[1]), "baseline": baseline,
               "best_ids": best}, open(f"{a.out}/summary.json", "w"), indent=2)

    # ---------------- summaries used in the report ----------------
    summ = {
        "by_activation": table.groupby("activation")[["train_acc", "val_acc", "val_macro_f1", "best_epoch"]].mean(),
        "by_optimizer": table.groupby("optimizer")[["train_acc", "val_acc", "val_macro_f1", "best_epoch"]].mean(),
        "by_arch": table.groupby("hidden")[["train_acc", "val_acc", "val_macro_f1", "best_epoch"]].mean(),
    }
    with open(f"{a.out}/summaries.txt", "w") as f:
        for k, v in summ.items():
            f.write(f"== {k} ==\n{v.round(4).to_string()}\n\n")

    # markdown copy of the tables
    with open(f"{a.out}/results_table.md", "w") as f:
        f.write(table.round(4).to_markdown(index=False) + "\n\n")
        f.write(test_table.round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Logistic regression baseline: {json.dumps(baseline)}\n")
    print("\nBest per activation (test set):\n", test_table.round(4).to_string(index=False))
    print("\nBaseline:", baseline)


if __name__ == "__main__":
    main()
