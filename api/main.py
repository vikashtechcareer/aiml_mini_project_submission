# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class TitanicInput(BaseModel):
    pclass: int
    sex: int  # 0 for female, 1 for male
    age: float
    sibsp: int
    parch: int
    fare: float

def load_model(dataset: str, model_name: str):
    path = f"models/{dataset}/{model_name}.pkl"
    return joblib.load(path)

@app.get("/")
def root():
    return {"message": "Welcome to the AIML Mini Project API"}

@app.post("/predict/iris/{model_name}")
def predict_iris(model_name: str, features: IrisInput):
    model = load_model("iris", model_name)
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/titanic/{model_name}")
def predict_titanic(model_name: str, features: TitanicInput):
    model = load_model("titanic", model_name)
    input_data = np.array([[features.pclass, features.sex, features.age, features.sibsp, features.parch, features.fare]])
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}
