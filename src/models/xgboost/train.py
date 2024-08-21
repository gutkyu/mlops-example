from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
import mlflow.xgboost
import numpy as np
import xgboost as xgb
import cloudpickle
import time

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}


def _train_with_hyperopt(X_train, y_train, X_val, y_val):
    def _train_model(params):
        # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.xgboost.autolog()
        with mlflow.start_run(nested=True):
            train = xgb.DMatrix(data=X_train, label=y_train)
            validation = xgb.DMatrix(data=X_val, label=y_val)
            # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
            # is no longer improving.
            booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                                evals=[(validation, "validation")], early_stopping_rounds=50)
            validation_predictions = booster.predict(validation)
            auc_score = roc_auc_score(y_val, validation_predictions)
            mlflow.log_metric('auc', auc_score)

            signature = infer_signature(X_train, booster.predict(train))
            mlflow.xgboost.log_model(booster, "model", signature=signature)
            
            # Set the loss to -1*auc_score so fmin maximizes the auc_score
            return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

    mlflow.xgboost.autolog()

    # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
    # A reasonable value for parallelism is the square root of max_evals.
    spark_trials = SparkTrials(parallelism=10)

    # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
    # run called "xgboost_models" .
    with mlflow.start_run(run_name='xgboost_models'):
        best_params = fmin(
            fn=_train_model, 
            space=search_space, 
            algo=tpe.suggest, 
            max_evals=96,
            trials=spark_trials,
        )

def train_model(X_train, y_train, X_val, y_val):
    _train_with_hyperopt(X_train, y_train, X_val, y_val)

def _compare_models():
    # TODO determine if the newly developed model performs better than the current production model
    pass

def evaluate_model():
    best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
    print(f'AUC of Best Run: {best_run["metrics.auc"]}')

    # _compare_models()

    return best_run.run_id

def register_model(model_name, run_id):
    new_model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)

    # Registering the model takes a few seconds, so add a small delay
    time.sleep(15)
    
    # 모델의 URI 확인
    model_uri = f"models:/{model_name}/{new_model_version.version}"
    print(f"Model URI: {model_uri}")

    # 모델의 버전 확인
    model_version = new_model_version.version
    print(f"Model Version: {model_version}")

    return model_uri, model_version
