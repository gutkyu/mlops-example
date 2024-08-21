import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]
    
        
def train_model(X_train, y_train, X_val, y_val):

    with mlflow.start_run(run_name='untuned_random_forest'):
        n_estimators = 10
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
        model.fit(X_train, y_train)

        # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
        predictions_test = model.predict_proba(X_val)[:,1]
        auc_score = roc_auc_score(y_val, predictions_test)
        mlflow.log_param('n_estimators', n_estimators)

        # Use the area under the ROC curve as a metric.
        mlflow.log_metric('auc', auc_score)
        wrappedModel = SklearnModelWrapper(model)

        # Log the model with a signature that defines the schema of the model's inputs and outputs. 
        # When the model is deployed, this signature will be used to validate inputs.
        signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
        
        # MLflow contains utilities to create a conda environment used to serve models.
        # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
        conda_env =  _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
                additional_conda_channels=None,
            )
        mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

def evaluate_model():
    baseline_run = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0]
    print(f'AUC of Baseline Run: {baseline_run["metrics.auc"]}')

    # _compare_models()

    return baseline_run.run_id


def register_model(model_name, run_id):
    
    new_model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

    # Registering the model takes a few seconds, so add a small delay
    time.sleep(15)


    # 모델의 URI 확인
    model_uri = f"models:/{model_name}/{new_model_version.version}"
    print(f"Model URI: {model_uri}")

    # 모델의 버전 확인
    model_version = new_model_version.version
    print(f"Model Version: {model_version}")

    return model_uri, model_version
