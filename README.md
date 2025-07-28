**Stacked Ensemble-Based Classification for Breast Cancer Diagnosis**
This project presents a robust machine learning pipeline for early breast cancer diagnosis using stacked ensemble learning, Boruta feature selection, and MLP optimization. Built on the widely-used Wisconsin Breast Cancer Dataset, it aims to classify tumors as benign or malignant with high accuracy.

--> :Key Features
1. Stacked Ensemble Architecture: Combines Random Forest, Gradient Boosting, and MLP as base learners with a Logistic Regression or HistGradientBoosting meta-classifier.

2.  Boruta Feature Selection: Reduces dimensionality by selecting 12 clinically relevant features.

3.  Preprocessing Pipeline: Includes imputation and standardization for clinical-grade data consistency.

4. 95–98%+ Accuracy across models with F1-scores nearing 0.98.

5. Streamlit Frontend: Real-time web application for healthcare practitioners to enter tumor metrics and get instant predictions.

--> Dataset :
Name: Wisconsin Breast Cancer Diagnostic Dataset

Source: UCI Machine Learning Repository

Attributes: 569 samples, 30 numeric features derived from FNA images, labeled as benign (357) or malignant (212).

--> Model Architecture :
1. Preprocessing
Mean imputation using SimpleImputer

Standardization using StandardScaler

2. Feature Selection
BorutaPy used with a Random Forest estimator to retain statistically significant features

3. Base Models
Random Forest (100 estimators)

Gradient Boosting Classifier

Multi-Layer Perceptron (MLP)

4. Meta Learner
Logistic Regression or HistGradientBoostingClassifier

Final classification using sigmoid output for binary prediction


--> Deployment :
A Streamlit app was built for real-time predictions:

Inputs: 12 features selected via Boruta

Output: Tumor classification (Benign or Malignant) with confidence

Frontend auto-loads model, imputer, scaler, and feature lis
