# Iris Flower Classification - Google Colab
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. Load built-in dataset (no CSV download required)
iris = load_iris(as_frame=True)
X = iris.data.copy()
y = iris.target.copy()
feature_names = list(X.columns)
target_names = iris.target_names
print("Dataset shape:", X.shape)
print("Classes:", list(target_names))
display(X.head())
display(X.describe())
print("Missing values:\n", X.isnull().sum())

# 2. Clean and split data. Stratification preserves class proportions.
X = X.replace([np.inf, -np.inf], np.nan)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))

# 3. Pipelines keep imputation/scaling fitted only on training data.
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)),
    "K-Nearest Neighbors": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=100, random_state=42))
}

# 4. Train and compare models
rows, trained, preds = [], {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name], preds[name] = model, pred
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Macro Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0)
    })
results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
display(results.round(3))

# 5. Accuracy chart
plt.figure(figsize=(8,4))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Test Accuracy by Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 6. Best model and detailed evaluation
best_name = results.iloc[0]["Model"]
best_model = trained[best_name]
best_pred = preds[best_name]
print("Best model:", best_name)
print(classification_report(
    y_test, best_pred, labels=np.arange(len(target_names)),
    target_names=target_names, zero_division=0
))

# 7. Confusion matrix
cm = confusion_matrix(y_test, best_pred, labels=np.arange(len(target_names)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 8. Predict one new sample (measurements are in centimetres)
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
predicted_class = best_model.predict(sample)[0]
print("Sample measurements:")
display(sample)
print("Predicted species:", target_names[predicted_class])
if hasattr(best_model, "predict_proba"):
    for flower, probability in zip(target_names, best_model.predict_proba(sample)[0]):
        print(f"{flower}: {probability:.3f}")

# 9. Automatically print a conclusion based on measured results
print("\nCONCLUSION")
print(f"Best test accuracy: {best_name} ({results.iloc[0]['Accuracy']*100:.2f}%)")
print(f"Correct predictions: {int(np.trace(cm))}/{int(cm.sum())}")
print("These results apply to this fixed test split; validate further before real-world use.")
