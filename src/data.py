import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from mlops_comm import CATALOG_NAME, SCHEMA_NAME, MODEL_NAME

CAT_PREFIX = f"{CATALOG_NAME}.{SCHEMA_NAME}.dataset"


def load_raw_data():
    
    # Read the white wine quality and red wine quality CSV datasets and merge them into a single DataFrame.
    
    white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
    red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

    return white_wine, red_wine

def preprocess(white_wine, red_wine):
    # Implement preprocessing steps here

    # Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

    red_wine['is_red'] = 1
    white_wine['is_red'] = 0
    
    df = pd.concat([red_wine, white_wine], axis=0)

    # Remove spaces from column names
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    #Looks like quality scores are normally distributed between 3 and 9.
    # Define a wine as high quality if it has quality >= 7.
    high_quality = (df.quality >= 7).astype(int)
    df.quality = high_quality

    # dropping null values, encoding categorical features, etc.
    df = df.dropna()

    # Additional preprocessing steps
    
    return df

def make_dataset(df):
    #Before training a model, check for missing values and split the data into training and validation sets.
    
    if df.isna().any().any():
        raise ValueError("Missing values exist in the DataFrame. Please handle the missing values before proceeding.")
    
    X = df.drop(["quality"], axis=1)
    y = df.quality

    # Split out the training data
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

    # Split the remaining data equally into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

    return X_train, y_train.to_frame(name="quality"), X_val,  y_val.to_frame(name="quality"), X_test, y_test.to_frame(name="quality")

def _reset_table(table_name):
     # 테이블 리셋
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    

def save_dataset(df_X_train, df_y_train, df_X_val, df_y_val, df_X_test, df_y_test):
    # 테이블 리셋
    _reset_table(f"{CAT_PREFIX}_X_train")
    _reset_table(f"{CAT_PREFIX}_y_train")
    _reset_table(f"{CAT_PREFIX}_X_val")
    _reset_table(f"{CAT_PREFIX}_y_val")
    _reset_table(f"{CAT_PREFIX}_X_test")
    _reset_table(f"{CAT_PREFIX}_y_test")

    spark = SparkSession.builder.getOrCreate()
    # 각 데이터셋을 PySpark DataFrame으로 변환
    X_train_spark = spark.createDataFrame(df_X_train)
    y_train_spark = spark.createDataFrame(df_y_train)
    X_val_spark = spark.createDataFrame(df_X_val)
    y_val_spark = spark.createDataFrame(df_y_val)
    X_test_spark = spark.createDataFrame(df_X_test)
    y_test_spark = spark.createDataFrame(df_y_test)
    
    # Unity Catalog에 데이터셋 저장
    
    X_train_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_X_train")
    y_train_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_y_train")
    X_val_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_X_val")
    y_val_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_y_val")
    X_test_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_X_test")
    y_test_spark.write.format("delta").saveAsTable(f"{CAT_PREFIX}_y_test")

def load_dataset():
    spark = SparkSession.builder.getOrCreate()
    
    # Load the datasets from Unity Catalog
    X_train = spark.table(f"{CAT_PREFIX}_X_train").toPandas()
    y_train = spark.table(f"{CAT_PREFIX}_y_train").toPandas()['quality']
    X_val = spark.table(f"{CAT_PREFIX}_X_val").toPandas()
    y_val = spark.table(f"{CAT_PREFIX}_y_val").toPandas()['quality']
    X_test = spark.table(f"{CAT_PREFIX}_X_test").toPandas()
    y_test = spark.table(f"{CAT_PREFIX}_y_test").toPandas()['quality']
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_test_data(X_train):
    # To simulate a new corpus of data, save the existing X_train data to a Delta table. 
    # In the real world, this would be a new batch of data.
    spark_df = spark.createDataFrame(X_train)

    table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data"

    (spark_df
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema",True)
        .saveAsTable(table_name)
    )

if __name__ == "__main__":
    white_wine, red_wine = load_raw_data()

    df = preprocess(white_wine, red_wine=red_wine)
    
    df_X_train, df_y_train, df_X_val, df_y_val, df_X_test, df_y_test = make_dataset(df)
    
    generate_test_data(df_X_train)

    save_dataset(df_X_train, df_y_train, df_X_val, df_y_val, df_X_test, df_y_test)