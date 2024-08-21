import mlflow.pyfunc
import numpy as np
import sklearn

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def load_model():
    # Load the MLflow model
    model_uri = "runs:/<RUN_ID>/random_forest_model"  # Replace <RUN_ID> with the actual run ID
    model = mlflow.pyfunc.load_model(model_uri)
    return SklearnModelWrapper(model)

def predict(data):
    model = load_model()
    predictions = model.predict(None, data)
    return predictions