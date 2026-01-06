import joblib
from train import models, preprocessor, X_train, y_train
from sklearn.pipeline import Pipeline

# Select best model
model = models["Random Forest"]

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/heart_disease_model.pkl")

print("Model saved successfully!")
