# DJSW_collabrative_coding

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![Documentation Status](https://readthedocs.org/projects/djsw-collabrative-coding/badge/?version=latest)](https://djsw-collabrative-coding.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/djsw-collabrative-coding.svg)](https://badge.fury.io/py/djsw-collabrative-coding)
[![Python version](https://img.shields.io/pypi/pyversions/djsw-collabrative-coding)](https://pypistats.org/packages/djsw-collabrative-coding)
[![License: MIT](https://img.shields.io/github/license/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/blob/master/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/issues)
[![Forks](https://img.shields.io/github/forks/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/network/members)
[![Stars](https://img.shields.io/github/stars/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding/commits/master)
[![Repo Size](https://img.shields.io/github/repo-size/youssefwally/DJSW_collabrative_coding)](https://github.com/youssefwally/DJSW_collabrative_coding)
<!-- ![tests](https://github.com/youssefwally/DJSW_collabrative_coding/actions/workflows/test_and_deploy.yml/badge.svg) -->


Efficient Lumi-friendly ML Pipeline

## Project Organization

```
├── LICENSE            <- MIT license.
│
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── datasetprep        <- Preparation for MNIST dataset.
│
├── docs               <- Sphix documentation.
│
├── models             <- Model Architectures.
│
├── reports            <- Generated slurm reports.
│
├── tests              <- Unit tests.
│
├── utils
│   ├──  dataset_mnist03_h5.py  <- Custom dataloader for MNIST (0-3).
│   ├──  mnist_dataset.py       <- Custom dataloader for MNIST (4-9).
│   └──  wdataloader.py         <- Custom dataloader for USPS (0-6).
│
├── weights            <- Weights for the different models.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`.
│
├── uv.lock            <- Lockfile used by uv to reproduce the analysis environment.
│
└── DJSW               <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes DJSW a Python module.
    │
    ├── evaluate.py             <- Code to evaluate trained models.
    │
    ├── run.sh                  <- Batch job script to be submitted to scheduler on LUMI.
    │
    ├── train.py                <- Code to train models.
    │
    └── main.py                 <- Main entrance script.
```

## CITATION
```bibtex
@software{DJSW2025,
  author       = {Dennis Adamek and Johannes Mørkrid and Sigurd Almli Hanssen and Youssef Wally},
  title        = {DJSW Collaborative Coding Template},
  year         = {2025},
  url          = {https://github.com/youssefwally/DJSW_collabrative_coding},
  version      = {1.0.0},
  license      = {MIT},
  note         = {GitHub repository},
}
```
## Results
| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score |
|-------|----------|-------------------|-----------|--------|----------|
| WMLP  | 0.9466   | 0.9381            | 0.9452    | 0.9381 | 0.9416   |
| DMLP  | 0.9957   | 0.9956            | 0.9956    | 0.9956 | 0.9956   |
| SMLP  | 0.9748   | 0.9750            | 0.9750    | 0.9750 | 0.9750   |

## Installation

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) or [uv](https://docs.astral.sh/uv/)
- Python 3.12 or higher

### Setup Instructions

You can set up this project using either **uv** or **conda**.

1. **Clone the repository**
    ```bash
    git clone git@github.com:<your-username>/DJSW_collaborative_coding.git
    cd DJSW_collaborative_coding
    ```
#### Using uv 

2. **Create and activate a Python virtual environment**
    ```bash
    uv venv 
    source .venv/bin/activate   # (Windows PowerShell: .venv\Scripts\Activate.ps1)
    ```

3. **Install all dependencies (runtime + dev) and create lockfile**
    ```bash
    uv sync --extra dev
    ```

4. **Install the package in editable (development) mode**
    ```bash
    uv pip install -e .
    ```

#### Using conda

2. **Create a conda environment**
    ```bash
    conda create -n djsw_env python=3.12
    conda activate djsw_env
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install the package in development mode** (optional)
    ```bash
    pip install -e .
    ```
#### Directly from Github using pip:
```bash
    pip install djsw@git+https://github.com/youssefwally/DJSW_collaborative_coding
```
### Verify Installation
```bash
python -c 'import DJSW; print("Installation successful!")'
```

## Run the code

There are three models available:
- `DMLP`: MLP with 3 hidden layers, 300 nodes per hidden layer and LeakyReLU activations
- `SMLP`: MLP with 4 hidden layers, 77 nodes per hidden layer and ReLU activations
- `WMLP`: MLP with 2 hidden layers, 100 nodes per hidden layer and LeakyReLU activations

There are also three corresponding dataset files and corresponding dataset loaders available:
- `mnist03.h5` with the `Mnist03Dataset` dataset class: MNIST data for numbers 0-3
- `mnist_4_9.h5` with the `MnistH5Dataset` dataset class: MNIST data for numbers 4-9
- `usps.h5` with the `USPS06Dataset` dataset class: USPS data

The following combinations of model and dataset are currently supported and can be accessed through the `--username` argument:
1) `DMLP` with `Mnist03Dataset`: choose `--username dennis`
2) `SMLP` with `MnistH5Dataset`: choose `--username sigurd`
3) `WMLP` with `USPS06Dataset`: choose `--username waly`

To train a model on the corresponding dataset run:
```
# uv
uv run DJSW/main.py --username name --exp_name train_name --output_dir ./weights/ --train --num_epochs 10 --batch_size 64 --lr 1e-3
# else
python DJSW/main.py --username name --exp_name train_name --output_dir ./weights/ --train --num_epochs 10 --batch_size 64 --lr 1e-3
```

To evaluate a trained model on the test split of the corresponding dataset run:
```
# uv
uv run DJSW/main.py --username name --exp_name test_name --output_dir . --load_checkpoint ./weights/train_name_checkpoint_epoch_10.pt
# else
python DJSW/main.py --username name --exp_name test_name --output_dir . --load_checkpoint ./weights/train_name_checkpoint_epoch_10.pt
```

## Individual reports
To view the individual reports see: [reports](https://youssefwally.github.io/DJSW_collaborative_coding/index.html).

--------
