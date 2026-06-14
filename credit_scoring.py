import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
# The source file is tab-delimited and does not contain an explicit target column.
df = pd.read_csv("dataset/credit_data.csv", sep="\t")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Create a binary target column for training because the raw file does not provide one.
median_amount = df["Credit amount"].median()
median_duration = df["Duration"].median()
df["creditworthy"] = (
    (df["Credit amount"] <= median_amount) & (df["Duration"] <= median_duration)
).astype(int)

# Target & features
X = df.drop(columns=["creditworthy"])
y = df["creditworthy"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess numeric and categorical columns
numeric_features = ["Age", "Job", "Credit amount", "Duration"]
categorical_features = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            numeric_features,
        ),
        (
            "cat",
            make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
            categorical_features,
        ),
    ]
)

# Model
model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=2000, solver="liblinear"),
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
