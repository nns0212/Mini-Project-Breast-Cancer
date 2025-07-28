import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from boruta import BorutaPy
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\BreastCancer\.venv\data.csv")
df.drop(columns=['Unnamed: 32', 'id'], errors='ignore', inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Handle missing data
X = X.dropna(axis=1, how='all')
X = X.fillna(X.mean())

# Boruta feature selection
rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=6, n_estimators=300, random_state=42)
boruta = BorutaPy(rf_boruta, n_estimators='auto', random_state=42)
boruta.fit(X.values, y.values)
selected_features = X.columns[boruta.support_].tolist()
X = X[selected_features]

# Preprocess
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(150, 75), max_iter=1500, random_state=42),
}

# Stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(150, 75), max_iter=1500, random_state=42))
]
meta_model = HistGradientBoostingClassifier(random_state=42)
models["Stacking"] = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, n_jobs=-1)

# Evaluation
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append([name, acc, prec, rec, f1, roc_auc])

# Format and display
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
pd.set_option("display.precision", 6)

print("\nSummary of All Models:\n")
print(results_df.to_string(index=False))

import matplotlib.pyplot as plt

# Plotting
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
x = np.arange(len(results_df["Model"]))  # the label locations
width = 0.15  # width of the bars

fig, ax = plt.subplots(figsize=(14, 6))

# Plot each metric with an offset
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric)

# Labels and formatting
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Evaluation Metrics")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df["Model"], rotation=45)
ax.set_ylim(0.9, 1.01)  # tighter y-axis since scores are all high
ax.legend()

plt.tight_layout()
plt.show()

