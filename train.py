"""Train and evaluate one configuration from the command line.

Example (best configuration, see README):
    python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 --test
"""
import argparse
import numpy as np
from nn import NeuralNet, load_data

def get_args():
    p = argparse.ArgumentParser(description="Neural network from scratch (NumPy)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64], help="e.g. --hidden 64 32")
    p.add_argument("--activation", choices=["sigmoid", "tanh", "relu"], default="relu")
    p.add_argument("--optimizer", choices=["sgd", "momentum", "rmsprop", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--l2", type=float, default=0.0, help="L2 weight decay strength")
    p.add_argument("--patience", type=int, default=15, help="early-stopping patience (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--test",  action="store_true", help="also report test-set results")
    return p.parse_args()


def main():
    a = get_args()
    np.random.seed(a.seed)
    net = NeuralNet(hidden=a.hidden, activation=a.activation, optimizer=a.optimizer,
                    lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, beta1=a.beta1,
                    beta2=a.beta2, l2=a.l2, patience=a.patience, seed=a.seed, verbose=True)
    data = net.preprocess(load_data(), split=tuple(a.split), seed=a.seed)
    print(f"classes: {data['classes']} | train/val/test sizes: "
          f"{len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])} |  "
          f"input features after preprocessing: {data['X_train'].shape[1]}")
    net.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    print(f"\nstopped after {net.epochs_run} epochs, best epoch {net.best_epoch}")
    for name in ("train", "val")+(("test",) if a.test else ()):
        r = net.evaluate(data[f"X_{name}"], data[f"y_{name}"])
        print(f"{name:5s} | loss {r['loss']:.4f} | accuracy {r['accuracy']:.4f} | macro-F1 {r['macro_f1']:.4f}")


if __name__   == "__main__":
    main()