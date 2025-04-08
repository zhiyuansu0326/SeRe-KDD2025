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

This repository uses six datasets as an example. They are:

- KuaiRec (kuairec)
- Yelp (yelp)
- MovieLens (movie)
- Facebook (facebook)
- Amazon - Video Games (vg)
- Amazon - Digital Music (dm)

 The names in brackets are the parameter names corresponding to the dataset in the code.The data loading functions are provided in `load_data.py`.

## Running Experiments

The experiment is executed via the `run.py` script. It supports command-line arguments for dataset selection, method, number of runs, and number of rounds per run. We set the default to run once and the number of rounds is 50000.

### Example Command

To run model A on dataset B for C runs and D rounds per time, execute:

```bash
python3 run.py --dataset "A" --method "B" --n_runs C --n_rounds D
```

For example, to run CLUB_N on KuaiRec for 5 runs and 20000 rounds per time:

```bash
python3 run.py --dataset "kuairec" --method "club_n" --n_runs 5 --n_rounds 20000
```

To run M-CNB_SeRe on Amazon - Video Games for 1 runs and 50000 rounds per time::

```bash
python3 run.py --dataset "vg" --method "mcnb_sere" --n_runs 1 --n_rounds 50000
```
