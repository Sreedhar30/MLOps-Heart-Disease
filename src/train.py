import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from preprocess import load_and_preprocess_data


DATA_PATH = "data/heart.csv"

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(DATA_PATH)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

mlflow.set_experiment("Heart Disease Classification")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log parameters
        mlflow.log_param("model_name", model_name)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"\n{model_name}")
        print(f"Accuracy : {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall   : {recall}")
        print(f"ROC-AUC  : {roc_auc}")

	# NEW — Save pipeline locally for reproducibility
        import os
        import joblib

        os.makedirs("models", exist_ok=True)
        model_filename = f"models/{model_name.replace(' ', '_').lower()}_pipeline.pkl"
        joblib.dump(pipeline, model_filename)

        print(f"Saved model pipeline to {model_filename}")
