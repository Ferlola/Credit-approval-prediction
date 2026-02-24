[Dataset info](https://archive.ics.uci.edu/dataset/27/credit+approval)

### 💳 Credit Approval Prediction
#### 📌 Project Overview

This project focuses on predicting credit approval. The objective is to build a robust classification model capable of accurately determining whether a credit application should be approved or rejected based on applicant data.

Two models were implemented and evaluated:
- Logistic Regression
- XGBoost Classifier (XGBClassifier)

Hyperparameter optimization was performed using Optuna to maximize model performance.

---

#### 🎯 Objective

The goal of this project is to:

- Develop predictive models for credit approval classification.
- Optimize model performance using automated hyperparameter tuning.
- Evaluate model stability and generalization through cross-validation.

---

#### 1️⃣ Models Implemented
**Logistic Regression**

Statistical model and a supervised machine learning algorithm used primarily for classification tasks.

**XGBClassifier**

An advanced gradient boosting model based on decision trees, designed for high predictive performance and efficiency.

#### 2️⃣ Hyperparameter Optimization

Hyperparameter tuning was conducted using **Optuna**, an automatic hyperparameter optimization framework.

For both models, Optuna searched for optimal configurations using cross-validation to ensure robust and unbiased performance estimates.

---

#### 📈 Results

The evaluation metric was based on cross-validation performance.

**Logistic Regression**

- Competitive optimization results.
- Stable cross-validation results.

**XGBClassifier**

- Achieved significantly better predictive performance.
- High mean cross-validation (CV) score.
- Low standard deviation across folds, indicating strong model stability and generalization capability.

The optimized XGBClassifier clearly outperformed Logistic Regression, demonstrating the power of gradient boosting methods in structured tabular data problems like credit approval prediction.

---

#### 🔍 Model Evaluation Strategy

- Stratified K-Fold Cross Validation
- Mean CV score used as the primary selection metric.
- Standard deviation of CV scores analyzed to evaluate model robustness.

---

#### 🚀 Key Takeaways

- Gradient boosting models can substantially outperform linear models in complex tabular datasets.
- Automated hyperparameter optimization (Optuna) significantly improves model performance.
- Cross-validation is essential to ensure model reliability and prevent overfitting.
- XGBClassifier achieved both high accuracy and strong stability across folds.

---

#### 📌 Conclusion

This project demonstrates the effectiveness of combining advanced machine learning algorithms with automated hyperparameter optimization to solve real-world financial classification problems. The XGBClassifier, optimized with Optuna, delivered outstanding and stable cross-validation performance, making it the preferred model for credit approval prediction.