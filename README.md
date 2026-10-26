# Assignment 3: Neural Network from Scratch

A feed-forward neural network written with NumPy only. It predicts whether a student will Drop out, stay Enrolled, or Graduate.

- **Dataset:** UCI *Predict Students' Dropout and Academic Success* (4,424 students, 36 features, 3 classes)
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Data is loaded from this raw URL**:
  https://raw.githubusercontent.com/SGcoder120/ML-assignment-3/refs/heads/main/data.csv
- **Random seed:** 42
- **Train / validation / test split:** 70% / 15% / 15%, stratified (3096 / 664 / 664 students)

## How to run

You need Python 3.9+ and an internet connection (the data is downloaded from the URL above).
Run the following commands in order:

```
# 1. Install the packages
pip install -r requirements.txt

# 2. Reproduce the best result
python train.py --hidden 64 32 --activation sigmoid --optimizer sgd --lr 0.1 --epochs 100 --batch_size 64 --patience 15 --seed 42 --test

# 3. Run all 24 experiments
python run_experiments.py --config experiments.json --out results
```

**Step 1** installs all of the requirements for this project.

**Step 2** should give a validation accuracy of 0.7711, a test accuracy of 0.7967, and a test macro-F1 of 0.7309.

**Step 3** takes about 30 seconds. It saves the results table, the best configuration per activation (with
test results), the learning curves, and the logistic regression baseline in the `results/` folder.

You can change any setting of `train.py` with these options: `--hidden` (one or two sizes, e.g. `64` or
`64 32`), `--activation` (sigmoid, tanh, relu), `--optimizer` (sgd, momentum, rmsprop, adam), `--lr`,
`--epochs`, `--batch_size`, `--beta1`, `--beta2`, `--l2`, `--patience`, `--seed`, `--split`, `--test`.

## Files

- `nn.py`: the `NeuralNet` class (`preprocess`, `train`, `predict`, `evaluate`)
- `train.py`: train and evaluate one configuration
- `run_experiments.py` and `experiments.json`: the 24-configuration experiment grid and its settings
- `requirements.txt`: packages to install
- `results/`, `results.pdf`, `report.pdf`: results table, learning curves, and the report
