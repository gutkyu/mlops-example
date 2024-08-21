import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# unit catalog name
CATALOG_NAME = os.getenv('CATALOG_NAME', 'databrickstest')
# unit catalog schema name
SCHEMA_NAME = os.getenv('SCHEMA_NAME', 'demo_mlops_dev')
# model name
MODEL_NAME = os.getenv('MODEL_NAME',f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_quality")

def find_model_version_by_alias(model_name, alias):
    
    client = MlflowClient()

    # model_version_infos = client.search_model_versions(f"name='{model_name}'")

    # # alias를 가진 모델 버전 찾기
    # alias_version = None
    # for model_version_info in model_version_infos:
    #     tags = model_version_info.tags
    #     if "alias" in tags and tags["alias"] == alias:
    #         alias_version = model_version_info.version
    #         break

    # if alias_version is not None:
    #     return alias_version
    # else:
    #     print(f"{alias} model version not found.")
    #     return None
    
    try:
        # 모델 버전 객체 반환
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        return model_version
    except MlflowException as e1:
        print(f"No model version found for alias '{alias}' in model '{model_name}'. Exception: {e1}")
        return None
    except Exception as e2:
        raise e2

