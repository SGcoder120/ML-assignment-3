# Assignment 2: Neural Network from Scratch (CS 4375)

Shivesh Gupta, NetID SXG220223

A feed-forward neural network (1 or 2 hidden layers) written in NumPy only. It predicts student
outcomes (Dropout / Enrolled / Graduate) on the UCI *Predict Students' Dropout and Academic Success*
dataset (4,424 students, 36 features).

- **Dataset page:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Data used by the code (raw URL, no local file needed):**
  https://raw.githubusercontent.com/SGcoder120/ML-assignment-3/refs/heads/main/data.csv
- **Seed:** 42 (split, weight initialization, shuffling)
- **Split:** 70 / 15 / 15 train / validation / test, stratified (3096 / 664 / 664)

## How to run

You need Python 3.9+ and an internet connection (the data is downloaded from the URL above).

**Step 1. Install the packages**
```
pip install -r requirements.txt
```

**Step 2. Reproduce the best result**
```
python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 --epochs 100 --batch_size 64 --patience 15 --seed 42 --test
```
Expected: validation accuracy 0.7711, test accuracy 0.7967, test macro-F1 0.7309.
(Best = highest validation macro-F1: sigmoid, hidden layers 64-32, plain SGD, lr 0.1.)

**Step 3. Run all 24 experiments** (about 30 seconds)
```
python run_experiments.py --config experiments.json --out results
```
This writes the results table, the best configuration per activation with test results, the learning
curves, and the logistic regression baseline into `results/`.

To try your own settings, change the options of `train.py`:
`--hidden` (one or two sizes, e.g. `64` or `64 32`), `--activation` (sigmoid, tanh, relu),
`--optimizer` (sgd, momentum, rmsprop, adam), `--lr`, `--epochs`, `--batch_size`, `--beta1`, `--beta2`,
`--l2`, `--patience`, `--seed`, `--split`, `--test`. The experiment grid is set in `experiments.json`.

## Files

| File | What it is |
|---|---|
| `nn.py` | The `NeuralNet` class (`preprocess`, `train`, `predict`, `evaluate`, forward pass, backprop, optimizers) |
| `train.py` | Train and evaluate one configuration |
| `run_experiments.py` | Runs the whole grid, picks the best per activation, makes plots and the baseline |
| `experiments.json` | Hyperparameters of the grid |
| `results/` | `results_table.csv` / `.md`, `best_per_activation.csv`, `learning_curves.png` (and one PNG per activation), `summary.json`, `summaries.txt` |
| `results.pdf` | Full results table, test results, learning curves |
| `report.pdf` | The 3-page report |

## What is implemented

- **Preprocessing:** missing values (median for numeric, most frequent for categorical), one-hot
  encoding of the 10 integer-coded categorical columns (listed in `CATEGORICAL` in `nn.py`),
  standardization of numeric columns. Everything is fitted on the training split only, giving 234 inputs.
- **Network:** fully vectorized (one matrix per mini-batch, loops only over layers), a bias in every
  hidden and output neuron, hidden activation sigmoid, tanh or ReLU.
- **Output layer:** softmax + cross-entropy for more than 2 classes (used here), sigmoid + binary
  cross-entropy for 2 classes.
- **Initialization:** He for ReLU hidden layers, Xavier/Glorot for sigmoid and tanh hidden layers. The
  output layer always uses Xavier because it feeds a sigmoid/softmax.
- **Optimizers:** plain mini-batch SGD, momentum, RMSProp, Adam. The experiments compare SGD with Adam.
- **Early stopping:** keeps the weights with the lowest validation loss (patience 15, max 100 epochs).

## Protocol and assumptions

- Tuning uses the validation set only. The test set is used once, for the best configuration of each
  activation, and for the logistic regression baseline (scikit-learn, `max_iter=3000`, same split).
- "Best" means highest validation macro-F1, since the classes are imbalanced.
- Each configuration is run once, so differences of about one percentage point are within noise.

## References

- Realinho, V., Vieira Martins, M., Machado, J., Baptista, L. (2021). *Predict Students' Dropout and
  Academic Success.* UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- S. Ruder, *An overview of gradient descent optimization algorithms*, https://arxiv.org/abs/1609.04747
- Goodfellow, Bengio and Courville, *Deep Learning*, Ch. 8, https://www.deeplearningbook.org
- Glorot and Bengio (2010), Xavier initialization; He et al. (2015), He initialization.
- AI assistance: prepared with help from Claude (Anthropic), as permitted by the instructor.