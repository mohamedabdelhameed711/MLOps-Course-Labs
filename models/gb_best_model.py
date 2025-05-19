import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

# 1 load data
CSV_PATH = r"D:\iti\MLOps\lab1_MLflow\MLOps-Course-Labs\dataset\Churn_Modelling.csv"         
df = pd.read_csv(CSV_PATH)

y = df["Exited"]
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])  # drop IDs

# 2 preprocessing pipeline
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

# 3 model & full pipeline
gb_params = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)

clf = GradientBoostingClassifier(**gb_params)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ]
)

# 4 train / test split  &  fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)

# 5 evaluation
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"GB Results  |  F1: {f1:.3f}  |  ROC-AUC: {auc:.3f}")

# 6 save model
os.makedirs("models", exist_ok=True)
MODEL_PATH = "gb_model.pkl"
joblib.dump(pipe, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
