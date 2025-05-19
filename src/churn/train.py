import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn.data import load_dataset
from churn.features import build_preprocessor
from churn.models import get_model


def train(
    model_name: str = "rf",
    model_params: dict | None = None,
    data_path: str = "dataset/Churn_Modelling.csv",
):
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    preprocessor = build_preprocessor(num_cols, cat_cols)

    model = get_model(model_name, **(model_params or {}))
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Bank-Churn-Prediction")

    with mlflow.start_run(run_name=f"{model_name}_run"):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "bank-churn.csv")   
        mlflow.log_param("Dataset", "bank-churn.csv")      


        pipeline.fit(X_train, y_train)

        # predict_proba or decision_function, whichever exists
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        y_pred = pipeline.predict(X_test)

        metrics = dict(
            roc_auc=roc_auc_score(y_test, y_prob),
            accuracy=accuracy_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
        )
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(
            f"  {model_name.upper()}  "
            f"AUC {metrics['roc_auc']:.3f}  "
            f"ACC {metrics['accuracy']:.3f}  "
            f"F1  {metrics['f1']:.3f}"
        )
