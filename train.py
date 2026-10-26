# Train and evaluate one configuration from the command line
# Example: python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 --test
import argparse

from nn import NeuralNet, load_data

def get_args():
    parser = argparse.ArgumentParser(description="Neural network from scratch (NumPy)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64], help="e.g. --hidden 64 32")
    parser.add_argument("--activation", choices=["sigmoid", "tanh", "relu"], default="relu")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "rmsprop", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--l2", type=float, default=0.0, help="L2 weight decay strength")
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--test", action="store_true", help="also report test-set results")
    return parser.parse_args()


def main():
    args = get_args()
    net = NeuralNet(hidden=args.hidden, activation=args.activation, optimizer=args.optimizer,
                    lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                    beta1=args.beta1, beta2=args.beta2, l2=args.l2,
                    patience=args.patience, seed=args.seed, verbose=True)
    data = net.preprocess(load_data(), split=tuple(args.split), seed=args.seed)
    print(f"classes: {data['classes']} | train/val/test sizes: "
          f"{len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])} | "
          f"input features after preprocessing: {data['X_train'].shape[1]}")

    net.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    print(f"\nstopped after {net.epochs_run} epochs, best epoch {net.best_epoch}")

    # The test set is only evaluated when --test is given
    splits = ["train", "val", "test"] if args.test else ["train", "val"]
    for name in splits:
        result = net.evaluate(data[f"X_{name}"], data[f"y_{name}"])
        print(f"{name:5s} | loss {result['loss']:.4f} | accuracy {result['accuracy']:.4f} "
              f"| macro-F1 {result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()