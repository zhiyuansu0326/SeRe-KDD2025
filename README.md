# Selective Reinitialization (SeRe)

This repository provides implementations of four Clustering of Neural Bandits (CNB) models and their SeRe-Enhanced versions, eight models in total. The available models are:

- **club_n**
- **locb_n**
- **sclub_n**
- **mcnb**
- **club_n_sere**
- **locb_n_sere**
- **sclub_n_sere**
- **mcnb_sere**

## Requirements

- Python 3.10.11
- PyTorch 2.0.1
- torchvision 0.15.2
- scikit-learn 1.5.2
- numpy 1.26.4
- scipy 1.13.1
- pandas 2.2.3

## Dataset

This repository uses MovieLens (movie) as an example.

## Running Experiments

The experiment is executed via the `run.py` script. It supports command-line arguments for dataset selection, method, number of runs, and number of rounds per run. We set the default to run once and the number of rounds is 50000.

### Example Command

To run model A on dataset B for C runs and D rounds per time, execute:

```bash
python3 run.py --dataset "A" --method "B" --n_runs C --n_rounds D
```

For example, to run M-CNB_SeRe on MovieLens for 1 run and 50000 rounds per time::

```bash
python3 run.py --dataset "movie" --method "mcnb_sere" --n_runs 1 --n_rounds 50000
```
