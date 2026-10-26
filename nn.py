# Feed-forward neural network from scratch (NumPy only) plus data preprocessing
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/SGcoder120/ML-assignment-3/refs/heads/main/data.csv"
TARGET = "Target"

# Integer-coded columns that are really categories
CATEGORICAL = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

# Each activation and its derivative (written in terms of the activation's output)
ACTIVATIONS = {
    "sigmoid": (expit, lambda a: a * (1 - a)),
    "tanh": (np.tanh, lambda a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)),
}


def load_data(url=None):
    # Read the semicolon-separated CSV straight from the raw GitHub URL
    df = pd.read_csv(url or DATA_URL, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def softmax(z):
    # Subtract the row max first so exp() cannot overflow
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(P, T):
    # Binary cross-entropy for one output column, softmax cross-entropy otherwise
    P = np.clip(P, 1e-12, 1 - 1e-12)
    if P.shape[1] == 1:
        return float(-np.mean(T * np.log(P) + (1 - T) * np.log(1 - P)))
    return float(-np.mean(np.sum(T * np.log(P), axis=1)))


# Fully-connected network with one or two hidden layers
class NeuralNet:
    def __init__(self, hidden=(64,), activation="relu", optimizer="adam",
                 lr=0.001, epochs=100, batch_size=64, beta1=0.9, beta2=0.999,
                 rho=0.9, l2=0.0, patience=15, seed=42, verbose=False):
        assert 1 <= len(hidden) <= 2, "use one or two hidden layers"
        assert activation in ACTIVATIONS
        assert optimizer in ("sgd", "momentum", "rmsprop", "adam")
        self.hidden = tuple(hidden)
        self.activation = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.optimizer = optimizer
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.beta1, self.beta2, self.rho = beta1, beta2, rho
        self.l2, self.patience, self.seed, self.verbose = l2, patience, seed, verbose
        self.params: dict = {}

    def preprocess(self, df, split=(0.70, 0.15, 0.15), seed=None):
        # Fill missing values, one-hot the categories, scale numbers, then split (stratified)
        seed = self.seed if seed is None else seed
        categorical = [c for c in CATEGORICAL if c in df.columns]
        numeric = [c for c in df.columns if c not in categorical + [TARGET]]

        labels = LabelEncoder()
        y = labels.fit_transform(df[TARGET].astype(str))
        X = df.drop(columns=[TARGET])

        _, val_size, test_size = split
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=val_size + test_size, stratify=y, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=test_size / (val_size + test_size),
            stratify=y_rest, random_state=seed)

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
        ])
        # Everything is fitted on the training set only
        return {
            "X_train": preprocessor.fit_transform(X_train),
            "X_val": preprocessor.transform(X_val),
            "X_test": preprocessor.transform(X_test),
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "classes": sorted(df[TARGET].astype(str).unique()),
        }

    def _init_params(self, n_in, n_out, rng):
        sizes = [n_in, *self.hidden, n_out]
        self.params = {}
        for l in range(len(sizes) - 1):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            is_hidden = l < len(sizes) - 2
            # He for ReLU hidden layers, Xavier for the rest (the output feeds a sigmoid or softmax)
            if is_hidden and self.activation == "relu":
                std = np.sqrt(2 / fan_in)
            else:
                std = np.sqrt(2 / (fan_in + fan_out))
            self.params[f"W{l}"] = rng.normal(0.0, std, size=(fan_in, fan_out))
            self.params[f"b{l}"] = np.zeros((1, fan_out))
        # Optimizer state
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _targets(self, y):
        if self.n_out == 1:
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.n_out)[y]

    def _forward(self, X):
        # A holds the input and every hidden layer's output
        A = [X]
        for l in range(len(self.hidden)):
            A.append(self.act(A[-1] @ self.params[f"W{l}"] + self.params[f"b{l}"]))
        last = len(self.hidden)
        z = A[-1] @ self.params[f"W{last}"] + self.params[f"b{last}"]
        P = expit(z) if self.n_out == 1 else softmax(z)
        return P, A

    def _backward(self, A, P, T):
        grads = {}
        # Softmax with cross-entropy and sigmoid with BCE both give (y - t) at the output
        dZ = (P - T) / len(T)
        for l in range(len(self.hidden), -1, -1):
            grads[f"W{l}"] = A[l].T @ dZ + self.l2 * self.params[f"W{l}"]
            grads[f"b{l}"] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dZ = (dZ @ self.params[f"W{l}"].T) * self.act_grad(A[l])
        return grads

    def _step(self, grads):
        self.t += 1
        for k, g in grads.items():
            p = self.params[k]
            if self.optimizer == "sgd":
                p -= self.lr * g
            elif self.optimizer == "momentum":
                self.m[k] = self.beta1 * self.m[k] + g
                p -= self.lr * self.m[k]
            elif self.optimizer == "rmsprop":
                self.v[k] = self.rho * self.v[k] + (1 - self.rho) * g ** 2
                p -= self.lr * g / (np.sqrt(self.v[k]) + 1e-8)
            else:
                # Adam with bias correction
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
                m_hat = self.m[k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[k] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def train(self, X_train, y_train, X_val, y_val):
        rng = np.random.default_rng(self.seed)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        self.n_out = 1 if n_classes == 2 else n_classes
        self._init_params(X_train.shape[1], self.n_out, rng)

        n = X_train.shape[0]
        T_train = self._targets(y_train)
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_loss, best_params, best_epoch, wait = np.inf, None, 0, 0

        for epoch in range(1, self.epochs + 1):
            # One pass over the training data in shuffled mini-batches
            order = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                batch = order[i:i + self.batch_size]
                P, A = self._forward(X_train[batch])
                self._step(self._backward(A, P, T_train[batch]))

            train, val = self.evaluate(X_train, y_train), self.evaluate(X_val, y_val)
            history["train_loss"].append(train["loss"])
            history["val_loss"].append(val["loss"])
            history["train_acc"].append(train["accuracy"])
            history["val_acc"].append(val["accuracy"])
            if self.verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"epoch {epoch:3d} | train loss {train['loss']:.4f} acc {train['accuracy']:.4f}"
                      f" | val loss {val['loss']:.4f} acc {val['accuracy']:.4f}")

            # Early stopping: remember the weights with the lowest validation loss
            if val["loss"] < best_loss - 1e-6:
                best_loss, best_epoch, wait = val["loss"], epoch, 0
                best_params = {k: v.copy() for k, v in self.params.items()}
            else:
                wait += 1
                if self.patience and wait >= self.patience:
                    break

        if best_params is not None:
            self.params = best_params
        self.history = history
        self.best_epoch = best_epoch
        self.epochs_run = len(history["train_loss"])
        return history

    def predict_proba(self, X):
        P, _ = self._forward(X)
        return np.hstack([1 - P, P]) if self.n_out == 1 else P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        P, _ = self._forward(X)
        pred = (P[:, 0] > 0.5).astype(int) if self.n_out == 1 else P.argmax(axis=1)
        return {"loss": cross_entropy(P, self._targets(y)),
                "accuracy": float(np.mean(pred == y)),
                "macro_f1": float(f1_score(y, pred, average="macro"))}