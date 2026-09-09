# Loan Approval Prediction

## Project Overview

This project develops a supervised machine-learning model to predict whether a loan application will be approved using applicant and loan-related information.

The project focuses on data preprocessing, class-imbalance handling, model comparison, evaluation, threshold selection, and business interpretation.

---

## Project Objective

The main objective is to build a reliable classification model that predicts loan approval using borrower features.

The project covers:

* Missing-value treatment
* Categorical-variable encoding
* Numerical-feature scaling
* Class-imbalance handling with SMOTE
* Comparison of linear and tree-based models
* Evaluation using precision, recall, F1-score, and ROC-AUC
* Business-oriented threshold selection
* Model interpretation and business recommendations

---

## Dataset Description

The dataset contains **614 loan applications** and **13 columns**, including the target variable.

Dataset source: [Loan Approval Prediction Case Study](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study)

| Column              | Description                              |
| ------------------- | ---------------------------------------- |
| `Loan_ID`           | Unique loan application identifier       |
| `Gender`            | Gender of the applicant                  |
| `Married`           | Applicant's marital status               |
| `Dependents`        | Number of dependents                     |
| `Education`         | Applicant's education level              |
| `Self_Employed`     | Whether the applicant is self-employed   |
| `ApplicantIncome`   | Monthly income of the applicant          |
| `CoapplicantIncome` | Monthly income of the co-applicant       |
| `LoanAmount`        | Requested loan amount                    |
| `Loan_Amount_Term`  | Loan repayment term                      |
| `Credit_History`    | Applicant's credit-history status        |
| `Property_Area`     | Urban, semiurban, or rural property area |
| `Loan_Status`       | Loan approval result                     |

The target variable is `Loan_Status`:

* `Y` represents an approved application.
* `N` represents a rejected application.

`Loan_ID` was excluded from model training because it is a unique identifier and does not contain meaningful predictive information.

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing values

* Missing numerical values were filled using the median.
* Missing categorical values were filled using the most frequent category.

Median imputation was selected because it is less affected by unusually high or low values.

### 2. Categorical encoding

Categorical variables were transformed using one-hot encoding.

One-hot encoding creates separate binary columns for every category without introducing an artificial numerical order.

### 3. Feature scaling

Numerical features were standardized using `StandardScaler`.

Scaling is especially important for Logistic Regression because it prevents variables with large numerical values from dominating variables with smaller values.

### 4. Train-test split

The dataset was divided into:

* 80% training data
* 20% testing data

A stratified split was used to preserve the original class distribution in both sets.

All preprocessing steps were included inside the modeling pipeline to reduce the risk of data leakage.

---

## Handling Class Imbalance with SMOTE

The dataset contains more approved applications than rejected applications. This imbalance could cause a model to focus mainly on the majority class.

SMOTE stands for **Synthetic Minority Oversampling Technique**. It creates synthetic observations for the minority class instead of simply copying existing records.

SMOTE was applied only to the training data and cross-validation training folds. It was not applied to the test set.

This approach helps the model learn minority-class patterns while keeping the final evaluation realistic.

---

## Models Compared

The following classification models were evaluated:

### Logistic Regression

A linear classification model that provides interpretable predictions and is relatively easy to explain and govern.

### Decision Tree

A rule-based model that can capture nonlinear relationships but may overfit small datasets.

### Random Forest

An ensemble of multiple decision trees that can capture complex relationships and interactions.

### Gradient Boosting

An ensemble method that builds trees sequentially, with each tree attempting to correct errors made by previous trees.

---

## Model Performance

The models were evaluated on the held-out test set using a classification threshold of 0.50.

| Model                   | Precision |    Recall |  F1-score |   ROC-AUC |
| ----------------------- | --------: | --------: | --------: | --------: |
| **Logistic Regression** | **0.862** | **0.882** | **0.872** | **0.876** |
| Random Forest           |     0.822 |     0.871 |     0.846 |     0.796 |
| Gradient Boosting       |     0.820 |     0.859 |     0.839 |     0.774 |
| Decision Tree           |     0.818 |     0.847 |     0.832 |     0.733 |

Logistic Regression achieved the best overall performance and was selected as the final model.

### Final model results

* **Precision:** 0.862
* **Recall:** 0.882
* **F1-score:** 0.872
* **ROC-AUC:** 0.876

### Metric interpretation

* **Precision:** The percentage of predicted approvals that were actually approved.
* **Recall:** The percentage of historically approved applications correctly identified.
* **F1-score:** The balance between precision and recall.
* **ROC-AUC:** The model's ability to rank approved applications above rejected applications across different thresholds.

---

## Recommended Threshold

The default classification threshold is 0.50. However, the best threshold depends on the business cost of incorrect decisions.

For this project, the threshold was selected using these assumptions:

* A false approval is five times more costly than a false rejection.
* Approval recall must remain at or above 70%.

The threshold was selected using out-of-fold predictions from the training data, not the test data.

The suggested threshold for controlled validation is **0.52**.

| Policy                    | Precision | Recall | F1-score | False approvals | False rejections |
| ------------------------- | --------: | -----: | -------: | --------------: | ---------------: |
| Default threshold: 0.50   |     0.862 |  0.882 |    0.872 |              12 |               10 |
| Suggested threshold: 0.52 |     0.869 |  0.859 |    0.864 |              11 |               12 |

Increasing the threshold from 0.50 to 0.52:

* Increased precision from 0.862 to 0.869
* Reduced false approvals from 12 to 11
* Reduced recall from 0.882 to 0.859
* Increased false rejections from 10 to 12

The 0.52 threshold is a slightly more conservative option. It should be treated as an initial validation threshold rather than a permanent production cutoff.

---

## Business Recommendations

* Use Logistic Regression as the preferred model because it achieved the best ROC-AUC and offers clearer explanations than the tree-based models.
* Begin controlled validation using a threshold of 0.52.
* Send applications close to the threshold for manual review instead of automatically approving or rejecting them.
* Verify missing credit-history information because Credit History was the strongest predictive feature.
* Review sensitive or proxy variables such as Gender and Married status for fairness and regulatory compliance.
* Validate the model on a larger and more recent dataset before production use.
* Calibrate model probabilities before interpreting the output as real approval or credit-risk probabilities.
* Monitor approval rates, defaults, manual overrides, fairness, and data drift after deployment.

The model should support loan officers and established lending policies rather than replace human judgement.

---

## Project Structure

```text
Loan-Approval-Prediction/
│
├── data/
│   └── loan_prediction.csv
│
├── notebook/
│   └── Alfido_Tech_Loan_Approval_Prediction.ipynb
│
├── report/
│   └── Loan_Approval_Prediction_Short_Report.pdf
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

---

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Replace `your-username` and `your-repository-name` with your actual GitHub username and repository name.

### Step 2: Create a virtual environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install the required packages

```bash
pip install -r requirements.txt
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open the notebook

Open the following file:

```text
notebook/Alfido_Tech_Loan_Approval_Prediction.ipynb
```

Restart the kernel and run all cells from beginning to end.

---

## Dataset Path

If the notebook is stored in the `notebook` folder and the CSV file is stored in the `data` folder, use:

```python
from pathlib import Path

candidate_paths = [
    Path("../data/loan_prediction.csv"),
    Path("data/loan_prediction.csv"),
    Path("loan_prediction.csv")
]

DATA_PATH = next(
    (path for path in candidate_paths if path.exists()),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "loan_prediction.csv was not found. Check the dataset location."
    )
```

---

## Requirements

Add the following packages to `requirements.txt`:

```text
pandas>=2.0,<3.0
numpy>=1.26,<3.0
matplotlib>=3.8,<4.0
seaborn>=0.13,<1.0
scikit-learn>=1.4,<1.9
imbalanced-learn>=0.12,<0.15
jupyter>=1.0,<2.0
ipykernel>=6.0,<7.0
```

---

## Limitations

* The dataset contains only 614 applications.
* The target represents historical approval decisions rather than actual repayment or default behaviour.
* Historical decisions may contain existing policy bias.
* The recommended threshold is based on an illustrative cost assumption.
* Predicted probabilities have not been calibrated for production use.
* Sensitive variables require fairness and regulatory review.
* Performance may change when applicant behaviour or economic conditions change.

This project should be treated as a proof of concept rather than a production lending system.

---

## Conclusion

This project successfully developed an end-to-end supervised machine-learning pipeline for loan approval prediction.

The workflow included missing-value treatment, one-hot encoding, feature scaling, SMOTE-based imbalance handling, model comparison, classification evaluation, feature interpretation, and cost-aware threshold selection.

Logistic Regression was selected as the final model after achieving:

* Precision of 0.862
* Recall of 0.882
* F1-score of 0.872
* ROC-AUC of 0.876

A threshold of 0.52 is recommended for controlled validation under the stated business assumptions.

Before production deployment, the model should be tested using larger and more recent data and reviewed for probability calibration, fairness, regulatory compliance, and financial impact.

---

## Author

**Kartik Kachwahe**

B.Tech in Information Technology
Aspiring Data Analyst and Data Scientist
