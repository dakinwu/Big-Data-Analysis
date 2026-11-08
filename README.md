# Critical Illness Insurance Purchase Prediction

A Python-based machine learning project that predicts whether existing customers will purchase critical illness insurance within the next three months.

## Overview

Identifying customers who may need additional insurance can help sales teams prioritize outreach. This project explores customer and policy data to build a binary classifier (a model that predicts one of two outcomes): purchase or no purchase.

The analysis covers data exploration, categorical encoding, class balancing, feature selection, model training, and prediction export.

## Methods

| Method | Purpose |
| --- | --- |
| Exploratory data analysis | Examine customer characteristics and relationships in the data. |
| Dummy encoding | Convert categorical values into numerical indicator columns. |
| SMOTE | Generate synthetic examples of the less common class to balance the training data. |
| Recursive Feature Elimination (RFE) | Repeatedly remove less useful features to select a smaller set of predictors. |
| Logistic regression | Estimate the likelihood of a yes-or-no outcome. |
| Statistical coefficient analysis | Examine the estimated relationships between selected features and purchase outcomes. |

SMOTE is applied after the initial training/test split so that synthetic samples are generated from the training portion.

## Workflow

1. Load the preprocessed customer data.
2. Explore feature distributions and class imbalance.
3. Encode categorical variables.
4. Split the data and balance the training portion.
5. Select features and inspect their statistical relationships.
6. Train a logistic regression model.
7. Evaluate predictions and export results.

## Repository Contents

| File | Description |
| --- | --- |
| `Data Exploration.ipynb` | Initial data exploration and visualization. |
| `Dummy SMOTE RFE.ipynb` | Main modeling notebook. |
| `train.zip`, `test.zip` | Original data archives. |
| `train_data.zip`, `test_data.zip` | Preprocessed data archives used by the analysis. |
| `layout.pdf` | Dataset field descriptions. |
| `國泰大數據.pdf` | Detailed project report in Chinese. |

## Tools

- Python
- Jupyter Notebook
- pandas and NumPy for data processing
- Matplotlib and seaborn for visualization
- scikit-learn for feature selection and classification
- imbalanced-learn for SMOTE
- statsmodels for statistical model summaries

## Getting Started

1. Obtain the repository files.
2. Extract the preprocessed archives and make `train_data.csv` and `test_data.csv` available in the notebook's working directory.
3. Prepare a Python environment with the packages listed above.
4. Review `Data Exploration.ipynb` for an introduction to the data.
5. Open `Dummy SMOTE RFE.ipynb` and review the cells in sequence.

The main notebook includes an export step that writes predictions to `Trial.csv`.

## Evaluation and Reproducibility

The notebook includes accuracy, precision, recall, F1-score (a balance between precision and recall), and ROC/AUC calculations (measures of class separation).

This repository preserves the original experimental notebooks. Package versions are not pinned, and older calls such as `fit_sample` may require compatibility adjustments.

Treat saved notebook outputs as historical results. For a new experiment, keep a final test set separate from resampling, feature selection, and model tuning.
