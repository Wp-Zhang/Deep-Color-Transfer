Deep Color Transfer
==============================

A short description of the project.

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
