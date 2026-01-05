from src.preprocess import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def test_model_training():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("data/heart.csv")

    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert score > 0.5
