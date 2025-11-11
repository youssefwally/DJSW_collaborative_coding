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
├── LICENSE            <- Open-source license
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project.
│
├── models             <- Model Architectures.
│
├── utils
│   ├── wdataloader.py <- Custom dataloader for USPS (0-6)
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── DJSW               <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── train.py                <- Code to train models
    │
    ├── evaluate.py                 <- Code to evaluate trained models
    │
    └── main.py                 <- Main entrance script
```

## CITATION
```bibtex
@software{DJSW2025,
  author       = {Dennis and Johannes Mørkrid and Sigurd Almli Hanssen and Youssef Wally},
  title        = {DJSW Collaborative Coding Template},
  year         = {2025},
  url          = {https://github.com/youssefwally/DJSW_collabrative_coding},
  version      = {1.0.0},
  license      = {MIT},
  note         = {GitHub repository},
}
```
## Results
| Model | Accuracy | Precision | Recall | Balanced Accuracy | F1 Score |
|-------|----------|-----------|--------|-------------------|----------|
| WMLP | 0.9466   | 0.9452    | 0.9381 | 0.9381            | 0.9416   |
| DMLP | 0.9916   | 0.9916    | 0.9914 | 0.9914            | 0.9915   |
| SMLP | 0.9748   | 0.9750    | 0.9750 | 0.9750            | 0.9750   |

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

## WMLP Documentation
### WMLP
1. Implemented a WMLP model, 2 hidden layers; 100 neurons each, with LeakyReLU as the activation function.
2. Model was trained by optimizing the Cross Entropy Loss as USPS is a multi-class classification task.
3. Model was evaluated with Balanced Accuracy as a evaluation metric to be robustly evaluate the model across different classes.
### wdataloader
1. Files are saved in an h5 file to save memory and be compatible with LUMI.
2. Images are loaded into memory from disk only when needed to not use up memory.
3. made an function that returns the input dim so that it can be passed directly to the model.
### Misc
1. Used CookieCutter to make boilerplate project outlet to follow worldwide code conventions
2. Everything went smoothly (Dennis)
3. Everything went smoothly (one typo as error)
4. Sphinx and LUMI
5. For such a simple project it was fairly easy to run jobs on LUMI. However, I would predict that for a large project with a rapidly changing environment it will be time consuming. (Job-IDs: 13618444, 13631080, 14077470)

--------

