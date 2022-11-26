Deep Color Transfer
==============================

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8e5c795af21f4f899f03095424f31179)](https://www.codacy.com/gh/Wp-Zhang/Deep-Color-Transfer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Wp-Zhang/Deep-Color-Transfer&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/8e5c795af21f4f899f03095424f31179)](https://www.codacy.com/gh/Wp-Zhang/Deep-Color-Transfer/dashboard?utm_source=github.com&utm_medium=referral&utm_content=Wp-Zhang/Deep-Color-Transfer&utm_campaign=Badge_Coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A course project of CS7150 at Northeastern University, MA.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |   ├── raw            <- Raw dataset directory
    |   |
    |   └── processed      <- Processed dataset directory, ready for model training
    │
    ├── docs               <- Project website
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   |
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to download or generate data
        │  
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        └── util           <- Scripts of tool functions


## Prepare dataset

We've uploaded the datasets to Kaggle: [MIT-Adobe 5K Dataset](https://www.kaggle.com/datasets/weipengzhang/adobe-fivek), [Segmentation Results](https://www.kaggle.com/datasets/weipengzhang/beit2-adobe5k).

1. Download these two datasets and extract them to `data/raw`, the data folder structure should be like this:
   ```
   data
   └── raw
      ├── adobe_5k
      |   ├── a
      |   ├── b
      |   ├── c
      |   ├── d
      |   ├── e
      |   └── raw
      └── segs
   ```

2. In the project root folder, use the provided script to prepare trainset
   ```
   python src/prepare_dataset.py
   ```
   The script will use the default config file. However, you can specify your own config file by adding `--config PATH_TO_CONFIG`

## Train the model
In the project root folder, run the following
```
python src/train.py
```
You can also specify your own config file by adding `--config PATH_TO_CONFIG` and change the *[Weights&Biases](https://wandb.ai/)* runner name by adding `--name NEW_NAME`.