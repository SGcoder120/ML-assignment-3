# Assignment 2 - Neural Network from Scratch (CS 4375)

Shivesh Gupta - NetID SXG220223

A feed-forward neural network (1 or 2 hidden layers) written with NumPy only, trained on the
UCI **Predict Students' Dropout and Academic Success** dataset (4,424 students, 36 features,
3 classes: Dropout / Enrolled / Graduate).

- Dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
  (the data is not included in this submission)
- Random seed: **42** (used for the split, weight initialization and mini-batch shuffling)
- Split: **70 / 15 / 15** train / validation / test, stratified by class
  (3096 / 664 / 664 students)

## Install

```
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```
Python 3.9+ is fine. Only NumPy is used for the network. pandas, scikit-learn and matplotlib are
used for data loading, preprocessing, splitting, metrics, the baseline and plots.

## Data loading

By default the data is downloaded with `ucimlrepo` (UCI id 697). If you are offline, download the
CSV from the link above and pass it with `--csv <path>` (semicolon separated). No path is hard-coded.

## Files

| File | Purpose |
|---|---|
| `nn.py` | `NeuralNet` class: `preprocess`, `train`, `predict`, `evaluate`, forward pass, backprop, optimizers |
| `train.py` | Train/evaluate one configuration from the command line |
| `run_experiments.py` | Runs the 24-configuration grid from `experiments.json`, picks the best per activation, test evaluation, learning curves, logistic regression baseline |
| `experiments.json` | All hyperparameters of the grid (activations, architectures, optimizers, learning rates, epochs, ...) |
| `test_gradcheck.py` | Compares backprop with numerical gradients (all activations, 2 and 3 classes, 1 and 2 hidden layers) |
| `results/` | `results_table.csv/.md`, `best_per_activation.csv`, `learning_curves.png` (+ one png per activation), `summary.json`, `summaries.txt` |
| `results.pdf` | Full results table, test results and learning curves |
| `report.pdf` | The 3-page report |

## Run

Single configuration (all hyperparameters are command-line arguments):

```
python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 \
                --epochs 100 --batch_size 64 --patience 15 --seed 42 --test
```
Options: `--hidden` (one or two sizes), `--activation {sigmoid,tanh,relu}`,
`--optimizer {sgd,momentum,rmsprop,adam}`, `--lr`, `--epochs`, `--batch_size`, `--beta1`, `--beta2`,
`--l2`, `--patience`, `--seed`, `--split`, `--csv`, `--test`.

Full experiment grid (24 configurations), tables, curves and baseline:

```
python run_experiments.py --config experiments.json --out results
```
(add `--csv data.csv` if you are using a local copy.) It takes about 30 seconds.

Gradient check: `python test_gradcheck.py` (all rows should say OK).

### Command that reproduces the best result

The best configuration (chosen on the validation set, macro-F1) is sigmoid, hidden layers 64-32,
plain SGD, learning rate 0.1:

```
python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 \
                --epochs 100 --batch_size 64 --patience 15 --seed 42 --test
```
Expected output: validation accuracy 0.7711, test accuracy 0.7967, test macro-F1 0.7309.

## What is implemented

- **Preprocessing** (`NeuralNet.preprocess`): missing values (median for numeric, most frequent
  for categorical), one-hot encoding of the 10 integer-coded categorical columns, standardization of
  the numeric columns. Imputers, encoder and scaler are fitted on the training split only and then
  applied to validation and test. 234 input features after encoding.
- **Network**: fully vectorized (a whole mini-batch is one matrix, loops only over layers), biases in
  every hidden and output neuron, hidden activation sigmoid / tanh / ReLU.
- **Output layer**: softmax + cross-entropy for more than two classes, sigmoid + binary
  cross-entropy for two classes (here: softmax, 3 classes).
- **Initialization**: He (normal, variance 2/fan_in) for ReLU hidden layers, Xavier/Glorot
  (normal, variance 2/(fan_in+fan_out)) for sigmoid and tanh hidden layers. The output layer feeds a
  sigmoid/softmax, so it always uses Xavier.
- **Optimizers**: plain mini-batch SGD, momentum, RMSProp and Adam. The experiments compare SGD
  against Adam.
- **Early stopping**: the weights with the lowest validation loss are kept (patience 15 epochs).

## Protocol and assumptions

- All tuning decisions use the validation set only. The test set is evaluated once, only for the best
  configuration of each activation.
- "Best" = highest validation macro-F1 (classes are imbalanced).
- Each configuration is run once with seed 42, so differences of about one percentage point are within noise.
- Training is capped at 100 epochs.
- The ten integer-coded columns treated as categorical are listed in `nn.py` (`CATEGORICAL`).
- The logistic regression baseline (scikit-learn, default settings, `max_iter=3000`) uses the same
  preprocessed split.

## References

- Realinho, V., Vieira Martins, M., Machado, J., Baptista, L. (2021). *Predict Students' Dropout and
  Academic Success.* UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- S. Ruder, *An overview of gradient descent optimization algorithms*, https://arxiv.org/abs/1609.04747
- Goodfellow, Bengio and Courville, *Deep Learning*, Ch. 8, https://www.deeplearningbook.org
- X. Glorot and Y. Bengio (2010), Xavier initialization; K. He et al. (2015), He initialization.
- AI assistance: this work was prepared with help from Claude (Anthropic), as permitted by the instructor.
