import pandas as pd
import numpy as np

# UCI Heart Disease (Cleveland) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert target to binary (0 = no disease, 1 = disease)
df["target"] = df["target"].astype(float)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

df.to_csv("data/heart.csv", index=False)

print("Dataset downloaded and saved to data/heart.csv")
