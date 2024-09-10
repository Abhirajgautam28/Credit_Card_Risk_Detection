# Credit Risk Detection

## Overview

This project focuses on detecting credit risk using various data science models and visualizations. The dataset used is `german_credit_data.csv`, which is stored in the `dataset` folder.

## Files and Folders

- **`dataset/german_credit_data.csv`**: The model training and evaluation dataset.
- **`outputs/using_matplotlib/`**: Contains visualizations created using Matplotlib.
- **`outputs/using_yellowbrick/`**: Contains visualizations created using Yellowbrick.

## Models and Visualizations

### Python Files

1. **`using_matplotlib.py`**:
   - **Models Used**: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest, Naive Bayes, Neural Network.
   - **Visualizations**: Includes various plots such as Confusion Matrix, ROC Curve, Precision-Recall Curve, and others using Matplotlib.

2. **`using_yellowbrick.py`**:
   - **Models Used**: K-Nearest Neighbors (KNN).
   - **Visualizations**: Includes ROC Curve, Precision-Recall Curve, and others using Yellowbrick.

## Installation

To run the code, first install the required packages. Use the following command:
```bash
pip install -r requirements.txt
