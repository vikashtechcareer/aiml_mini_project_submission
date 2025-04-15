# train_models.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def train_and_save_models(X, y, dataset_name):
    models = {
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(f"models/{dataset_name}", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{dataset_name}/{name}.pkl")
        print(f"{name} trained and saved for {dataset_name}")

# Iris Dataset
def process_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    train_and_save_models(X, y, "iris")

# Titanic Dataset
def process_titanic():
    df = pd.read_csv("data/titanic.csv")

    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

    # Fill missing age
    df["Age"] = SimpleImputer(strategy="mean").fit_transform(df[["Age"]])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    train_and_save_models(X, y, "titanic")

if __name__ == "__main__":
    process_iris()
    process_titanic()
