
![logo](docs/assets/img/front-logo.png)
==============================

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8e5c795af21f4f899f03095424f31179)](https://www.codacy.com/gh/Wp-Zhang/Deep-Color-Transfer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Wp-Zhang/Deep-Color-Transfer&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/8e5c795af21f4f899f03095424f31179)](https://www.codacy.com/gh/Wp-Zhang/Deep-Color-Transfer/dashboard?utm_source=github.com&utm_medium=referral&utm_content=Wp-Zhang/Deep-Color-Transfer&utm_campaign=Badge_Coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Color transfer between images based on deep learning.

Project Organization
------------

    ├── LICENSE
    ├── README.md
    │
    ├── data               <- Data directory
    │
    ├── docs               <- Project website
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Referenced materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the experiments.
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to process data.
        │  
        ├── models         <- Scripts to construct models
        │
        └── util           <- Scripts of tool functions

## Environment setup
1. Install required packages by `pip install -r requirements.txt`
2. We used Weights & Biases to do experiment tracking, you can install it by `pip install wandb`
3. Init Weights & Biases by `wandb init`

## Prepare dataset

We've uploaded the datasets to Kaggle: [MIT-Adobe 5K Dataset](https://www.kaggle.com/datasets/weipengzhang/adobe-fivek), [Segmentation Results](https://www.kaggle.com/datasets/weipengzhang/beit2-adobe5k).

1. Download these two datasets and extract them to `data/raw`, the data folder structure should be like this:
   ```
   data
   └── raw
      ├── adobe_5k
      │   ├── a
      │   ├── b
      │   ├── c
      │   ├── d
      │   ├── e
      │   └── raw
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

## Model Inference
You can put the test data anywhere you want, but the folder structure should be like this:
```
   data
   ├── in_imgs   <- Input images in .jpg format
   ├── in_segs   <- Segmentation results of input images in .npy format
   ├── ref_imgs  <- Reference images in .jpg format
   └── ref_segs  <- Segmentation results of reference images in .npy format
```
In the project root folder, run the following
```
python src/inference.py --config PATH_TO_MODEL_CONFIG --weights PATH_TO_MODEL_STAET_DICT --test_dir PATH_TO_TEST_DATA
```

If your images are too large and raise an out-of-memory error, or you want to speed up the inference process, you can resize the input images by adding `--resize_dim TARGET_DIM[int]`

If you also want to pass the semantic segmentation results of the images to the model, you can add `--use_seg`

The generated images will be saved under the test data folder.

## Knowledge Distillation

We found that knowledge distillation can speed up model convergence and get a smaller model with similar performance. To do this, you can run:
```
python src/train_distill.py --config PATH_TO_CONFIG --teacher-weights PATH_TO_TEACHER_STATE_DICT
```
**Some tips for training with knowledge distillation**: we found that changing the loss weights, soft loss weights, and identical pair loss weight can help with model convergence. Here's what we did to train a smaller model with similar performance:
| Epochs | loss_lambda0 | loss_lambda1 | loss_lambda2 | soft_loss_weight |
|--------|--------------|--------------|--------------|------------------|
| 0-9    | 0.5          | 10           | 10           | 0.5              |
| 10-18  | 0.5          | 10           | 10           | 0                |
| 19-24  | 1            | 1.5          | 0.5          | 0                |