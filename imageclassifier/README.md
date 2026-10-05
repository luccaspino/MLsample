# imageclassifier

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

image classifier

The Python package in `ic/` contains the reusable logic behind the image,
TensorFlow teaching, and cardiovascular notebooks. The notebooks are kept as
orchestration and explanation layers; data preparation, models, training, and
evaluation live in Python modules.

## Setup

```bash
uv sync
```

## Command line

```bash
uv run imageclassifier train-cifar10
uv run imageclassifier train-heart --path /path/to/heart.csv
```

Models are saved in `models/` and best checkpoints are saved in
`models/checkpoints/`. The CIFAR-10 workflow keeps data augmentation disabled
and uses the same CNN architecture as the original notebook.

## Project Organization

```text
imageclassifier/
├── data/                    <- raw, interim, processed, and external data
├── models/                  <- serialized models and checkpoints
├── notebooks/               <- exploratory and teaching notebooks
├── reports/figures/         <- generated plots
├── tests/                   <- pytest tests for data and modeling contracts
└── ic/
    ├── config.py            <- project paths and typed hyperparameters
    ├── dataset.py           <- public dataset loading facade
    ├── features.py          <- tabular feature preparation
    ├── main.py              <- Typer command-line application
    ├── plots.py              <- reusable visualizations
    └── modeling/
        ├── pipeline.py      <- model factories
        ├── split.py         <- train/validation splitting
        ├── train.py         <- training and artifact metadata
        ├── evaluate.py      <- metrics and thresholding
        └── predict.py       <- model loading and inference
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ic and configuration for tools like black
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
└── ic   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ic a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

