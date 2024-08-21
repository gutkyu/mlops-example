from pyspark.sql.functions import struct
import mlflow
from mlops_comm import CATALOG_NAME, SCHEMA_NAME, MODEL_NAME

def inference(model_name):
    # 모델 UDF 생성
    apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@Champion")

    table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data"
    # Unity Catalog 테이블에서 새로운 데이터 읽기
    new_data = spark.read.table(table_name)

    # 모델을 새로운 데이터에 적용
    print(new_data.columns)
    udf_inputs = struct(*(new_data.columns))
    new_data = new_data.withColumn("prediction", apply_model_udf(udf_inputs))

    # 예측 결과 컬럼을 추가한 new_data를 다시 테이블에 반영, 새로운 스키마로 기존 테이블 덮어쓰기 
    new_data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

    print("예측 결과 반영 : 완료")

if __name__ == "__main__":
    inference(MODEL_NAME)