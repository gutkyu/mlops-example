from mlflow.tracking import MlflowClient

def _extract_model_name_version(model_uri):
    model_name, model_version = model_uri.split('/')[-2:]
    return model_name, model_version
    
def _get_model_version(model_name, model_version):
    client = MlflowClient()
    return client.get_model_version(
        name=model_name,
        version=model_version
    )

def _set_model_validation_status(model_version, status):
    # status : PENDING, PASSED, FAILED
    client = MlflowClient()
    # "model_validation_status" 태그를 "PENDING"으로 설정
    client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="model_validation_status",
        value=status
    )

def _check_format_metadata(model_version):
    # basic format and metadata validations
    pass

def _check_performance_evaluations(model_version):
    # performance evaluations on selected data slices,
    pass

def _check_compliance(model_version):
    # compliance with organizational requirements such as compliance checks for tags or documentation.
    pass

def _check_pre_deploy(model_version):
    pass
     
def _update_challenger(model_version):
    
    client = MlflowClient()
    client.set_registered_model_alias(model_version.name, "Challenger", model_version.version)
    
    print("Assign the Challenger alias to the new version")

def validate(model_uri):
    model_name, model_version = _extract_model_name_version(model_uri)
    model_version = _get_model_version(model_name, model_version)
    _set_model_validation_status(model_version=model_version, status="PENDING")
    try:
        _check_compliance(model_version)
        _check_pre_deploy(model_version)
    except:
        _set_model_validation_status(model_version=model_version, status="FAILED")
    else:
        _set_model_validation_status(model_version=model_version, status="PASSED")
    _update_challenger(model_version)

if __name__ == "__main__":
    # train.py 실행 출력 마지막 줄을 업데이트한다.
    model_uri = "models:/databrickstest.demo_mlops_dev.wine_quality/6"
    validate(model_uri)