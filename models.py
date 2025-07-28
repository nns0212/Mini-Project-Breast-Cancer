# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Load data
df = pd.read_csv("C:\\Users\\ASUS\\PycharmProjects\\BreastCancer\\.venv\\data.csv")
df.drop(columns=['Unnamed: 32', 'id'], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Feature selection using Boruta
rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, n_estimators=100, random_state=42)
boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', random_state=42, max_iter=50)
boruta_selector.fit(X.values, y.values)
selected_features = X.columns[boruta_selector.support_]
X = X[selected_features]
print(f"\nSelected Features ({len(selected_features)}): {list(selected_features)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save selected features and scaler
joblib.dump(selected_features.tolist(), 'selected_features.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
}


# Train and evaluate
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # Save model
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.joblib")

# Summary Table
results_df = pd.DataFrame(results).set_index("Model")
print("\nSummary of All Models:\n")
print(results_df)

# Bar plot for comparison
results_df.plot(kind='bar', figsize=(12, 6))
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ROC Curve Comparison
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of All Models")
plt.legend()
plt.grid()
plt.show()


