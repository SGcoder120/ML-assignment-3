"""Feed-forward neural network from scratch (NumPy only) + data preprocessing.

Dataset: UCI "Predict Students' Dropout and Academic Success" (id 697).
No deep-learning or autodiff library is used. sklearn is only used for data
loading, preprocessing, splitting and metrics, as allowed by the assignment.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

UCI_ID = 697
TARGET = "Target"
# Integer-coded columns that are really categories (see the UCI variable table).
CATEGORICAL = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]


def load_data(csv=None):
    """Load the dataset from a CSV path (if given) or from the UCI repository."""
    if csv:
        with open(csv, "r", encoding="utf-8-sig") as f:
            header = f.readline()
        sep = ";" if header.count(";") > header.count(",") else ","
        df = pd.read_csv(csv, sep=sep)
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=UCI_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [c.strip() for c in df.columns]
    return df


class NeuralNet:
    """Fully-connected network with one or two hidden layers.

    Hidden activation: sigmoid | tanh | relu.
    Output: sigmoid + binary cross-entropy (2 classes) or softmax + cross-entropy.
    Optimizer: sgd | momentum | rmsprop | adam (all mini-batch).
    """

    def __init__(self, hidden=(64,), activation="relu", optimizer="adam",
                 lr=0.001, epochs=100, batch_size=64, beta1=0.9, beta2=0.999,
                 rho=0.9, l2=0.0, patience=15, seed=42, verbose=False):
        assert 1 <= len(hidden) <= 2, "use one or two hidden layers"
        assert activation in ("sigmoid", "tanh", "relu")
        assert optimizer in ("sgd", "momentum", "rmsprop", "adam")
        self.hidden = tuple(hidden)
        self.activation = activation
        self.optimizer = optimizer
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.beta1, self.beta2, self.rho = beta1, beta2, rho
        self.l2, self.patience, self.seed, self.verbose = l2, patience, seed, verbose
        self.params = None

    # ------------------------------------------------------------------
    # Preprocessing and splitting
    # ------------------------------------------------------------------
    def preprocess(self, df, target=TARGET, categorical=None,
                   split=(0.70, 0.15, 0.15), seed=None):
        """Clean the data, split it (stratified) and scale using train only.

        Missing values: median (numeric) / most frequent (categorical).
        Categorical columns: one-hot. Numeric columns: standardized.
        All transformers are fitted on the training split only.
        """
        seed = self.seed if seed is None else seed
        categorical = CATEGORICAL if categorical is None else categorical
        categorical = [c for c in categorical if c in df.columns]
        numeric = [c for c in df.columns if c not in categorical + [target]]

        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        X = df.drop(columns=[target])

        tr, va, te = split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=va + te, stratify=y, random_state=seed)
        X_va, X_te, y_va, y_te = train_test_split(
            X_tmp, y_tmp, test_size=te / (va + te), stratify=y_tmp, random_state=seed)

        self.preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
        ])
        out = {
            "X_train": self.preprocessor.fit_transform(X_tr),   # fit on train only
            "X_val": self.preprocessor.transform(X_va),
            "X_test": self.preprocessor.transform(X_te),
            "y_train": y_tr, "y_val": y_va, "y_test": y_te,
            "classes": list(le.classes_),
        }
        return out

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------
    def _act(self, z):
        if self.activation == "sigmoid":
            return expit(z)
        if self.activation == "tanh":
            return np.tanh(z)
        return np.maximum(0.0, z)

    def _act_grad(self, a, z):
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        if self.activation == "tanh":
            return 1.0 - a ** 2
        return (z > 0).astype(z.dtype)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _init_params(self, n_in, n_out, rng):
        """Xavier/Glorot for sigmoid/tanh hidden layers, He for ReLU hidden layers.
        The output layer feeds a sigmoid/softmax, so it always uses Xavier."""
        sizes = [n_in, *self.hidden, n_out]
        self.params = {}
        n_layers = len(sizes) - 1
        for l in range(n_layers):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            is_hidden = l < n_layers - 1
            if is_hidden and self.activation == "relu":
                std = np.sqrt(2.0 / fan_in)                 # He
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))     # Xavier
            self.params[f"W{l}"] = rng.normal(0.0, std, size=(fan_in, fan_out))
            self.params[f"b{l}"] = np.zeros((1, fan_out))
        self.n_layers = n_layers
        # optimizer state
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    # ------------------------------------------------------------------
    # Forward / loss / backward (all vectorized over the mini-batch)
    # ------------------------------------------------------------------
    def _forward(self, X):
        A, Zs = [X], []
        h = X
        for l in range(self.n_layers - 1):
            z = h @ self.params[f"W{l}"] + self.params[f"b{l}"]
            h = self._act(z)
            Zs.append(z)
            A.append(h)
        L = self.n_layers - 1
        z = h @ self.params[f"W{L}"] + self.params[f"b{L}"]
        if self.n_out == 1:                      # sigmoid output
            P = expit(z)
        else:                                    # softmax output (stable)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            P = e / e.sum(axis=1, keepdims=True)
        return P, A, Zs

    def _targets(self, y):
        if self.n_out == 1:
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.n_out)[y]

    @staticmethod
    def _loss_from(P, T, binary):
        P = np.clip(P, 1e-12, 1 - 1e-12)
        if binary:
            return float(-np.mean(T * np.log(P) + (1 - T) * np.log(1 - P)))
        return float(-np.mean(np.sum(T * np.log(P), axis=1)))

    def _backward(self, A, Zs, P, T):
        m = T.shape[0]
        grads = {}
        dZ = (P - T) / m              # softmax+CE and sigmoid+BCE both give y - t
        for l in range(self.n_layers - 1, -1, -1):
            grads[f"W{l}"] = A[l].T @ dZ + self.l2 * self.params[f"W{l}"]
            grads[f"b{l}"] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dA = dZ @ self.params[f"W{l}"].T
                dZ = dA * self._act_grad(A[l], Zs[l - 1])
        return grads

    def loss_and_grads(self, X, y):
        """Loss (+ L2 penalty) and gradients for a batch; used by the gradient check."""
        P, A, Zs = self._forward(X)
        T = self._targets(y)
        loss = self._loss_from(P, T, self.n_out == 1)
        if self.l2:
            loss += 0.5 * self.l2 * sum(np.sum(self.params[f"W{l}"] ** 2)
                                        for l in range(self.n_layers))
        return loss, self._backward(A, Zs, P, T)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
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
            else:  # adam
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
                m_hat = self.m[k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[k] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ------------------------------------------------------------------
    # Training / prediction / evaluation
    # ------------------------------------------------------------------
    def train(self, X_train, y_train, X_val, y_val):
        """Mini-batch training. Keeps the weights with the lowest validation loss
        (early stopping with `patience` epochs)."""
        rng = np.random.default_rng(self.seed)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        self.n_classes = n_classes
        self.n_out = 1 if n_classes == 2 else n_classes
        self._init_params(X_train.shape[1], self.n_out, rng)

        n = X_train.shape[0]
        T_train = self._targets(y_train)
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_loss, best_params, best_epoch, wait = np.inf, None, 0, 0

        for ep in range(1, self.epochs + 1):
            idx = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                b = idx[i:i + self.batch_size]
                P, A, Zs = self._forward(X_train[b])
                self._step(self._backward(A, Zs, P, T_train[b]))

            tr = self.evaluate(X_train, y_train)
            va = self.evaluate(X_val, y_val)
            hist["train_loss"].append(tr["loss"]); hist["val_loss"].append(va["loss"])
            hist["train_acc"].append(tr["accuracy"]); hist["val_acc"].append(va["accuracy"])
            if self.verbose and (ep % 10 == 0 or ep == 1):
                print(f"epoch {ep:3d} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f}"
                      f" | val loss {va['loss']:.4f} acc {va['accuracy']:.4f}")

            if va["loss"] < best_loss - 1e-6:
                best_loss, best_epoch, wait = va["loss"], ep, 0
                best_params = {k: v.copy() for k, v in self.params.items()}
            else:
                wait += 1
                if self.patience and wait >= self.patience:
                    break

        self.params = best_params            # restore best weights
        self.history = hist
        self.best_epoch = best_epoch
        self.epochs_run = len(hist["train_loss"])
        return hist

    def predict_proba(self, X):
        P, _, _ = self._forward(X)
        return np.hstack([1 - P, P]) if self.n_out == 1 else P

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y):
        P, _, _ = self._forward(X)
        loss = self._loss_from(P, self._targets(y), self.n_out == 1)
        pred = (P[:, 0] > 0.5).astype(int) if self.n_out == 1 else P.argmax(axis=1)
        return {"loss": loss,
                "accuracy": float(np.mean(pred == y)),
                "macro_f1": float(f1_score(y, pred, average="macro"))}
