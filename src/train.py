import mlflow
from mlflow.tracking import MlflowClient
from mlops_comm import CATALOG_NAME, SCHEMA_NAME, MODEL_NAME, find_model_version_by_alias
from data import load_dataset
from models.sklearn.train import train_model as skl_train_model, evaluate_model as skl_evaluate_model, register_model as skl_register_model
from models.xgboost.train import train_model as xgb_train_model, evaluate_model as xgb_evaluate_model, register_model as xgb_register_model

def _load_baseline_model(model_name):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")

    # Baseline 모델 버전 찾기
    baseline_version = None
    for model_version in model_versions:
        tags = model_version.tags
        if "baseline" in tags and tags["baseline"] == "true":
            baseline_version = model_version.version
            break
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version.version
        )
        tags = model_version_details.tags
        print(tags)
    else:
        print("Baseline model not found.")
        return None

    baseline_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{baseline_version}")
    return baseline_model

def train_baseline_model(X_train, y_train, X_val, y_val):
    found_champion = find_model_version_by_alias(MODEL_NAME, alias="Champion") is not None
    if found_champion:
        print("Skipping baseline model training: A champion model already exists.")
        return
    skl_train_model(X_train, y_train, X_val, y_val)
    run_id = skl_evaluate_model()
    model_uri, model_version = skl_register_model(MODEL_NAME, run_id)
    client = MlflowClient()
    client.set_model_version_tag(name=MODEL_NAME, version=model_version, key="baseline", value="true")
    print(f"baseline model 생성, model_uri={model_uri}, model_version={model_version}")

    model_version_details = client.get_model_version(
        name=MODEL_NAME,
        version=model_version
    )
    tags = model_version_details.tags
    print(tags)

    mdl = _load_baseline_model(MODEL_NAME)
    return model_uri, model_version

def train_challenger_model(X_train, y_train, X_val, y_val):
    skl_train_model(X_train, y_train, X_val, y_val)
    run_id = skl_evaluate_model()
    model_uri, model_version = skl_register_model(MODEL_NAME, run_id)
    print(f"challenger model 생성, model_uri={model_uri}, model_version={model_version}")
    return model_uri, model_version

if __name__ == "__main__":
    # Load the training data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    # Train and log the baseline model
    train_baseline_model(X_train, y_train, X_val, y_val)

    # Train and log the challenger model
    model_uri, model_version = train_challenger_model(X_train, y_train, X_val, y_val)

    print(f"Challenger model uri : {model_uri}")
