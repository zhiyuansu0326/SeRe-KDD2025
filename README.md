# Selective Reinitialization (SeRe)

This repository provides implementations of 4 Clustering of Neural Bandits (CNB) models and their SeRe-Enhanced version, 8 models in total. The available models are:

- **club_n**
- **locb_n**
- **sclub_n**
- **mcnb**
- **club_n_sere**
- **locb_n_sere**
- **sclub_n_sere**
- **mcnb_sere**

## Requirements

- Python 3.9.6
- PyTorch 2.0.1
- torchvision 0.15.2
- scikit-learn 1.5.2
- numpy 1.26.4
- scipy 1.13.1
- pandas 2.2.3

## Dataset

This repository uses the **MovieLens** dataset as an example. The data loading function is provided in `load_data.py`.

## Running Experiments

The experiment is executed via the `run.py` script. It supports command-line arguments for dataset selection, method, number of runs, and number of rounds per run. We set the default to run once and the number of rounds is 10000.

### Example Command

To run the `mcnb_sere` model on the MovieLens dataset, execute:

```bash
python3 run.py --dataset "movielens" --method "mcnb_sere"
```
