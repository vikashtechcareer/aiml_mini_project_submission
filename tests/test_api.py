import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

# --- Fixtures ---

@pytest.fixture
def iris_valid_inputs():
    return [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
        {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}
    ]


@pytest.fixture
def titanic_valid_inputs():
    return [
        {"pclass": 3, "sex": 1, "age": 22.0, "sibsp": 1, "parch": 0, "fare": 7.25},
        {"pclass": 1, "sex": 0, "age": 38.0, "sibsp": 1, "parch": 0, "fare": 71.2833},
        {"pclass": 2, "sex": 0, "age": 27.0, "sibsp": 0, "parch": 0, "fare": 13.0}
    ]


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_iris_valid(iris_valid_inputs, model_name):
    for features in iris_valid_inputs:
        res = requests.post(f"{BASE_URL}/predict/iris/{model_name}", json=features)
        assert res.status_code == 200
        assert "prediction" in res.json()


@pytest.mark.parametrize("model_name", ["decision_tree", "random_forest", "gradient_boosting"])
def test_titanic_valid(titanic_valid_inputs, model_name):
    for features in titanic_valid_inputs:
        res = requests.post(f"{BASE_URL}/predict/titanic/{model_name}", json=features)
        assert res.status_code == 200
        assert "prediction" in res.json()

