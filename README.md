# Malicious URL Detection using Random Forest and Other Models

## Overview
This project explores the use of machine learning models to detect and classify malicious URLs. The dataset consists of URLs labeled as **benign, defacement, malware, and phishing**.

Initially, several models were trained and evaluated, but after comparing their performance, **Random Forest** was selected as the final model due to its superior performance.

## Project Structure

- **src/all_models.py**: Contains the code where several models (Logistic Regression, Decision Tree, Random Forest, SVM, and Gradient Boosting) were evaluated and compared.
- **src/random_forest.py**: The final, optimized code that uses only the Random Forest model for detection.
- **data/processed_dataset.csv**: The processed dataset used for training and testing the models.
- **results/random_forest_confusion_matrix.png**: Confusion matrix of the Random Forest model.
- **notebooks/model_comparison.ipynb**: (Optional) Jupyter Notebook containing model comparison and visualizations.

## How to Use

### Step 1: Clone the Repository
```bash
git clone <repo-url>
