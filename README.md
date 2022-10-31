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
    │   ├── train          <- Trainset
    |   |   ├── input
    |   |   |   ├── imgs   <- Input images
    |   |   |   └── segs   <- Input image semantic segmentation results
    |   |   └── reference
    |   |       ├── imgs   <- Reference images
    |   |       └── segs   <- Reference image semantic segmentation results
    |   |
    │   └── test           <- Testset
    |       ├── input
    |       |   ├── imgs   <- Input images
    |       |   └── segs   <- Input image semantic segmentation results
    |       └── reference
    |           ├── imgs   <- Reference images
    |           └── segs   <- Reference image semantic segmentation results
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │  
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │  
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        ├── util           <- Scripts of tool functions
        │ 
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
