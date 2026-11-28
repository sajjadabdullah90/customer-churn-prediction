# Customer Churn Prediction Using Machine Learning

A supervised machine learning project that predicts whether a telecommunications customer is likely to leave the company using the IBM Telco Customer Churn dataset.

**Project Repository:** [sajjadabdullah90/customer-churn-prediction](https://github.com/sajjadabdullah90/customer-churn-prediction)  
**Main Notebook:** [Customer Churn Prediction Notebook](notebook/Customer_Churn_Prediction.ipynb)  
**Interactive Dashboard:** Streamlit application (`app.py`)

---

## 1. Project Overview

Customer churn occurs when a customer stops using a company's services. Identifying customers who are likely to leave can help businesses understand retention risks and take preventive action.

This project explores customer demographics, account information, subscribed services, and billing details to build and evaluate binary classification models for churn prediction.

| Project detail | Description |
|---|---|
| Problem type | Binary classification |
| Learning type | Supervised machine learning |
| Domain | Telecommunications |
| Dataset | IBM Telco Customer Churn |
| Dataset size | 7,043 customer records |
| Target variable | `Churn` |
| Target classes | `Yes` (churn) and `No` (no churn) |
| Models evaluated | Logistic Regression and Random Forest |

## 2. Problem Statement

Predict whether an existing telecommunications customer is likely to churn using customer demographics, account details, subscribed services, and billing information, so that the business can identify at-risk customers and support customer-retention decisions.

## 3. Project Objectives

- Explore the dataset and identify patterns associated with customer churn.
- Assess data quality and handle missing or incorrectly typed values.
- Engineer useful features from existing customer information.
- Prepare numerical and categorical data for machine learning.
- Establish a majority-class baseline for comparison.
- Train and compare Logistic Regression and Random Forest models.
- Evaluate performance using cross-validation and held-out test data.
- Examine classification errors, especially missed churn cases.

## 4. Dataset

The project uses the IBM Telco Customer Churn dataset, which contains 7,043 customer records and information about services, contracts, account tenure, and charges.

The target column is `Churn`:

- `Yes`: the customer left the company.
- `No`: the customer remained with the company.

The dataset has an imbalanced target distribution:

| Churn class | Records | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

**Dataset file:** [`dataset/Telco-Customer-Churn.csv`](dataset/Telco-Customer-Churn.csv)

The dashboard reads the CSV directly from the repository's `dataset/` directory. When running the notebook, make sure its working directory or CSV path points to this file.

## 5. Workflow

The notebook follows an end-to-end supervised learning workflow:

1. Import libraries and load the dataset.
2. Inspect the dataset structure, data types, missing values, and duplicates.
3. Analyze the target distribution.
4. Correct data types and handle missing values.
5. Perform exploratory data analysis (EDA).
6. Analyze categorical and numerical features in relation to churn.
7. Engineer additional customer-behaviour features.
8. Select the model input features and target variable.
9. Split the data into training and testing sets using stratification.
10. Build preprocessing pipelines for numerical and categorical features.
11. Train a majority-class baseline.
12. Train and compare Logistic Regression and Random Forest.
13. Evaluate models using cross-validation and test-set metrics.
14. Review confusion matrices, classification reports, and prediction errors.

## 6. Feature Engineering and Selection

Two additional features are created:

| Feature | Description |
|---|---|
| `TotalServices` | Counts the selected services marked as `Yes` for each customer. |
| `TenureGroup` | Groups customer tenure into four ranges: 0–12, 13–24, 25–48, and 49–72 months. |

The following columns are excluded from the model input:

- `customerID`: unique customer identifier rather than a behavioural predictor.
- `gender`: excluded after initial exploratory analysis.
- `PhoneService`: its service information is partly represented in the engineered `TotalServices` feature.

The target variable `Churn` is kept separately. The final feature matrix contains **19 input features**, including the two engineered features.

## 7. Data Preprocessing

Numerical and categorical columns are processed separately using a Scikit-learn `ColumnTransformer` and `Pipeline`.

| Feature type | Preprocessing |
|---|---|
| Numerical | Median imputation and standardization |
| Categorical | Most-frequent imputation and one-hot encoding |

Keeping preprocessing inside the model pipelines ensures that transformations are fitted on training data during model evaluation, helping reduce data leakage.

The dataset is split into **80% training data and 20% testing data**, using stratification to preserve the target-class distribution.

## 8. Models

Three approaches are evaluated:

- **Dummy Classifier:** majority-class baseline that predicts the most frequent class.
- **Logistic Regression:** linear classification model used for binary prediction.
- **Random Forest:** ensemble model that combines predictions from multiple decision trees.

Both machine learning models use the same preprocessing pipeline for a fair comparison.

## 9. Model Evaluation

### Cross-validation results

Five-fold stratified cross-validation is performed on the training data.

| Model | Mean CV Accuracy | Standard Deviation |
|---|---:|---:|
| Logistic Regression | **80.32%** | 1.26% |
| Random Forest | 79.00% | 1.08% |
| Baseline | 73.46% | 0.01% |

Logistic Regression achieved the highest mean cross-validation accuracy among the evaluated models.

### Test-set results

The following metrics are calculated on the held-out test set.

| Model | Accuracy | Churn Precision | Churn Recall |
|---|---:|---:|---:|
| Logistic Regression | **80.55%** | **65.72%** | **55.88%** |
| Random Forest | 78.07% | 61.09% | 47.86% |
| Baseline | 73.46% | 0.00% | 0.00% |

Logistic Regression performed better than Random Forest on the reported test metrics. It identified 55.88% of actual churn cases, compared with 47.86% for Random Forest.

> **Important:** Accuracy alone does not tell the full story for this imbalanced dataset. Churn-class precision, recall, F1-score, and the confusion matrix are also considered.

## 10. Error Analysis

The selected Logistic Regression model produced **274 incorrect predictions** on the test set:

- **109 false positives:** customers predicted to churn who actually stayed.
- **165 false negatives:** customers predicted to stay who actually churned.

False negatives are particularly important because they represent customers who may leave without being identified by the model.

## 11. Key Findings

- The dataset contains 7,043 customer records, with churn representing 26.54% of observations.
- Contract type, tenure, internet service, and payment method show notable associations with churn in the exploratory analysis.
- Logistic Regression achieved 80.32% mean cross-validation accuracy.
- On the held-out test set, Logistic Regression achieved 80.55% accuracy and 55.88% churn recall.
- Both models performed better than the majority-class baseline, but identifying churn customers remains a challenge.

## 12. Technologies and Libraries

- **Python**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** and **Seaborn** — data visualization
- **Scikit-learn** — preprocessing, pipelines, model training, cross-validation, and evaluation
- **Jupyter Notebook** — development and experimentation

## 13. How to Run the Project

### Prerequisites

- Python 3.9 or later
- pip
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sajjadabdullah90/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Place `Telco-Customer-Churn.csv` in the notebook's working directory.

4. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `notebook/Customer_Churn_Prediction.ipynb` and run the cells in order.

## 14. Interactive Analytics Dashboard

The Streamlit dashboard is a visualization and inference layer over the notebook outputs. **The notebook is the single source of truth for preprocessing, training, cross-validation, and evaluation.** The dashboard does not retrain models or independently recreate the evaluation results.

### Dashboard sections

- **Executive Overview:** customer base, observed churn rate, selected-model accuracy and churn recall, retention mix, and model scorecard.
- **Customer Insights:** churn rates by customer segment, tenure and billing distributions, and numerical feature correlations.
- **Churn Predictor:** use the exact fitted Logistic Regression or Random Forest pipeline exported by the notebook.
- **Model Performance:** notebook test metrics, cross-validation results, confusion matrices, classification reports, and ROC curves.
- **Error Analysis:** inspect false positives and false negatives from the notebook's selected Logistic Regression model.
- **Data Explorer:** filter the original customer records and export a selected segment.

### Run the notebook and export artifacts first

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open a terminal in the repository's `dataset/` directory (the notebook reads `Telco-Customer-Churn.csv` relative to its working directory):

   ```bash
   cd dataset
   jupyter notebook ../notebook/Customer_Churn_Prediction.ipynb
   ```

3. Run every notebook cell from top to bottom, including **Section 20: Exporting Results for the Interactive Dashboard**. This creates the `artifacts/` directory at the repository root, containing the fitted model pipelines and evaluation outputs.

### Start the dashboard

From the repository root, run:

```bash
streamlit run app.py
```

If the dashboard reports missing artifacts, complete Section 20 in the notebook first and restart Streamlit. The app loads saved results and models; it does not repeat training or cross-validation.

## 15. Future Improvements

- Improve recall for the churn class through decision-threshold tuning.
- Experiment with class-weighting and resampling techniques.
- Explore additional feature engineering strategies.
- Evaluate further classification algorithms.
- Use precision-recall curves and additional metrics to guide model selection.
- Validate performance on new or unseen customer data.

## 16. Conclusion

This project demonstrates an end-to-end machine learning workflow for telecommunications customer churn prediction, from exploratory data analysis and feature engineering to model evaluation and error analysis.

Based on the current evaluation, **Logistic Regression is the best-performing model among the two machine learning models tested**. However, its churn recall of 55.88% shows that further improvement is needed before the model could be considered reliable for real-world customer-retention decisions.

---

*Developed as a machine learning project for the AI Engineering Bootcamp.*
