import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from mlops_comm import MODEL_NAME, find_model_version_by_alias
from data import load_dataset

def _load_test_data():
    # Load the dataset
    _, _, _, _, X_test, y_test = load_dataset()
    return X_test, y_test

def _load_baseline_model(model_name):
    client = MlflowClient()
                                              
    model_versions = client.search_model_versions(f"name='{model_name}'")

    # Baseline 모델 버전 찾기
    baseline_version = None
    for model_version in model_versions:
        # search_model_versions() 반환 항목 중 model_version.tags 항목은 항상 빈값이다.
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version.version
        )
        tags = model_version_details.tags
        if "baseline" in tags and tags["baseline"] == "true":
            baseline_version = model_version.version
            break
    else:
        print("Baseline model not found.")
        return None

    baseline_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{baseline_version}")
    return baseline_model

def compare_challenger_champion(model_name):
    # "Challenger" 모델 확인
    challenger_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")

    # "Champion" 모델 확인
    target_model = None
    champion_version = find_model_version_by_alias(model_name=model_name, alias="Champion")

    found_champion = champion_version is not None
    if found_champion:
        # "Champion" 모델 로드
        target_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
        if target_model is None:
            raise ValueError("champion model 적재 실패했습니다.")
    else:
        # baseline 모델 로드
        print("champion model을 찾을 수 없습니다. baseline 모델을 비교합니다.")
        target_model = _load_baseline_model(model_name)
        if target_model is None:
            raise ValueError("baseline model을 찾을 수 없습니다.")

    # 평가 데이터셋 로드 (예시로 X_test, y_test 사용)
    X_test, y_test = _load_test_data()

    # 두 모델의 성능 평가
    target_model_auc = roc_auc_score(y_test, target_model.predict(X_test))
    challenger_auc = roc_auc_score(y_test, challenger_model.predict(X_test))
    

    # 성능 결과 기록
    with mlflow.start_run():
        if found_champion:
            print(f'AUC Champion: {target_model_auc}')
            mlflow.log_metric("champion_auc", target_model_auc)
        else:
            print(f'AUC Baseline: {target_model_auc}')
            mlflow.log_metric("baseline_auc", target_model_auc)
        print(f'AUC Challenger: {challenger_auc}')
        mlflow.log_metric("challenger_auc", challenger_auc)

    # 성능 비교 
    if challenger_auc < target_model_auc:
        if found_champion:
            raise ValueError("baseline model보다 성능이 낮습니다.")
        raise ValueError("champion model보다 성능이 낮습니다.")
    print("Challenger model outperformed Champion.")
    
def update_champion(model_name):
    if model_name is None:
        raise ValueError("champion으로 지정할 challenger model_name을 전달해주세요.")
    challenger_version = find_model_version_by_alias(model_name=model_name, alias="Challenger")
    model_version = challenger_version.version
    client = MlflowClient()
    # Assign the "Champion" alias to the new version.
    client.set_registered_model_alias(model_name, "Champion", model_version)
    print("Challenger is now the new Champion.")
    

def deploy(model_name):
    # batch inference 배포는 inference에서 직접 challenge model 적재
    pass

if __name__ == "__main__":
    compare_challenger_champion(MODEL_NAME)
    update_champion(MODEL_NAME)
    deploy(MODEL_NAME)


   