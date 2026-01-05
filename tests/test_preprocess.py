import pandas as pd
from src.preprocess import load_and_preprocess_data

def test_preprocess_output_shapes():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("data/heart.csv")

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert preprocessor is not None
