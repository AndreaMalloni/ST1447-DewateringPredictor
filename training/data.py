from datetime import datetime
from .logger import LoggerManager
import os
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, row_number, when
from pyspark.ml.feature import VectorAssembler
from typing import Tuple
from pyspark.sql import Window
from pyspark.ml.feature import StandardScaler


__all__ = ["process_data", "to_numpy_array", "to_pyspark_vector", "CONFIG_DIR"]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

def process_data(df: DataFrame, config: str) -> DataFrame:
    logger = LoggerManager.get_logger("Processing", enabled=config["logging"])

    try:
        logger.info("Starting data processing")

        if not config["full_dataset"]:
            logger.info("Filtering data based on params specfied in the configuration.")
            df = df.filter(col("nome_parametro").isin(config["params"]))

        if config["dataset_partitioning"]["sample"] is not 1:
            df = df.sample(
                fraction=config["dataset_partitioning"]["sample"], 
                seed=config["dataset_partitioning"]["seed"])
            
        df = df.na.drop("all")

        window_spec = Window.partitionBy("nome_parametro").orderBy("data_registrazione")
        df = df.withColumn("row_id", row_number().over(window_spec))

        logger.info("Pivoting and aggregating data by 'nome_parametro'.")
        df = df.select("row_id", "nome_parametro", "valore") \
            .groupBy("row_id") \
            .pivot("nome_parametro") \
            .agg({"valore": "first"}) \
            .cache()
        df = df.drop("row_id")
        
        logger.info("Data grouping to identify duplicates with count > 1.")
        df = df.groupBy(df.columns).count().filter("count > 1")
        df = df.dropDuplicates()

        if config["logging"]: 
            for column in df.columns:
                logger.info(df.select(column).describe().toPandas().to_string())   

        df = df.drop("count")

        return df

    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise


def to_numpy_array(df: DataFrame, features_col: str, label_col: str, config: str) -> Tuple[np.ndarray, np.ndarray]:
    logger = LoggerManager.get_logger("Processing", enabled=config["logging"])

    try:
        logger.info("Converting DataFrame to NumPy arrays.")
        features = np.array(df.select(features_col).rdd.map(lambda row: row[0]).collect())
        labels = np.array(df.select(label_col).rdd.map(lambda row: row[0]).collect())
        logger.info("Conversion to NumPy arrays completed successfully.")
        return features, labels
    except Exception as e:
        logger.error(f"Error during conversion to NumPy arrays: {e}")
        raise

def to_pyspark_vector(df: DataFrame, config: str) -> Tuple[DataFrame, DataFrame]:
    logger = LoggerManager.get_logger("Processing", enabled=config["logging"])

    try:
        logger.info("Initializing VectorAssembler for feature transformation.")
        
        feature_columns = config["params"] if config["params"] else df.columns.copy()
        if config["prediction_feature"] in feature_columns:
            feature_columns.remove(config["prediction_feature"])
        
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features", handleInvalid="skip")
        data = assembler.transform(df)

        if config.get("feature_scaling", False):  # Controlla se lo scaling è abilitato nel config
            logger.info("Applying StandardScaler to features.")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
            data = scaler.fit(data).transform(data)
        else:
            data = data.withColumnRenamed("raw_features", "features")

        logger.info("Feature transformation completed successfully.")

        final_data = data.select("features", config["prediction_feature"])

        logger.info("Splitting data into training and testing sets.")
        train_data, test_data = final_data.randomSplit(
            [config["dataset_partitioning"]["train"], config["dataset_partitioning"]["test"]], 
            seed=config["dataset_partitioning"]["seed"]
        )
        logger.info("Data split completed successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error during data transformation: {e}")
        raise

if __name__ == "__main__":
    pass