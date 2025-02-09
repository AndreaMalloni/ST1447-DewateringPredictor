#!./venv/bin/python

import argparse
from datetime import datetime
import json
import os, shutil
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import tensorflow as tf
from training.data import *
from training.logger import LoggerManager


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def linear_regression(train_data: DataFrame, test_data: DataFrame, outputLabel: str, config: str) -> None:
    logger = LoggerManager.get_logger(f'[Training] {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', enabled=config["logging"])

    try:
        logger.info("Initializing Linear Regression model training.")
        lr = LinearRegression(featuresCol="features", labelCol=outputLabel, predictionCol="Predicted_PV16_TorbiditaChiarificato", regParam=0.01)
        lr_model = lr.fit(train_data)

        logger.info("Linear Regression model training completed.")
        predictions = lr_model.transform(test_data)

        logger.info("Evaluating Linear Regression model.")
        evaluator = RegressionEvaluator(labelCol=outputLabel, predictionCol="Predicted_PV16_TorbiditaChiarificato", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Root Mean Squared Error (RMSE) on test data: {rmse:.3f}")

        evaluator_r2 = RegressionEvaluator(labelCol=outputLabel, predictionCol="Predicted_PV16_TorbiditaChiarificato", metricName="r2")
        r2 = evaluator_r2.evaluate(predictions)
        logger.info(f"R-squared (R2) on test data: {r2:.3f}")
        return lr_model
    except Exception as e:
        logger.error(f"Error during Linear Regression training: {e}")
        raise

def neural_network_regression(train_data: DataFrame, test_data: DataFrame, outputLabel: str, config: str) -> None:
    logger = LoggerManager.get_logger(f'[Training] {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', enabled=config["logging"])

    try:
        logger.info("Converting train and test data to NumPy arrays for neural network training.")
        train_features, train_labels = to_numpy_array(train_data, "features", outputLabel, config)
        test_features, test_labels = to_numpy_array(test_data, "features", outputLabel, config)

        logger.info("Initializing Neural Network model training.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse',  # Mean Squared Error
                      metrics=['mae'])  # Mean Absolute Error

        model.fit(train_features, train_labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        logger.info("Neural Network model training completed.")

        logger.info("Evaluating Neural Network model.")
        test_loss, test_mae = model.evaluate(test_features, test_labels, verbose=0)
        logger.info(f"Test Loss (MSE): {test_loss:.3f}")
        logger.info(f"Test MAE: {test_mae:.3f}")

        predictions = model.predict(test_features)
        ss_total = np.sum((test_labels - np.mean(test_labels)) ** 2)
        ss_residual = np.sum((test_labels - predictions.flatten()) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        logger.info(f"R-squared (R2) on test data: {r2:.3f}")
        return model
    except Exception as e:
        logger.error(f"Error during Neural Network training: {e}")
        raise

def valid_config(name: str) -> str:
    if name not in os.listdir(CONFIG_DIR):
        raise argparse.ArgumentTypeError(f"Error: The provided training config does not exist")
    return name

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for data processing and model training.")
    parser.add_argument('-c', '--config', 
                        type=valid_config, 
                        required=True, 
                        choices=[config for config in os.listdir(CONFIG_DIR)], 
                        help='Filename of the config to use')
    parser.add_argument('-o', '--out', 
                        type=str, 
                        required=False, 
                        help='Name of the output model file')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()

        with open(os.path.join(CONFIG_DIR, args.config)) as configfile:
            config = json.load(configfile)

            logger = LoggerManager.get_logger(f'[Training] {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', enabled=config["logging"])
            logger.info(f"Loaded training configuration: {args.config}")
            
            spark = SparkSession.builder \
            .appName("Linear Regression with PySpark MLlib") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .getOrCreate()
            logger.info("Spark session started successfully.")

            logger.info("Reading and preprocessing the dataset.")
            df: DataFrame = spark.read.option("encoding", "UTF-8").csv(config["source"], header=True, inferSchema=True)
            df = df.repartition(100)
            df = process_data(df, config)

            logger.info("Transforming data into feature vectors.")
            train_data, test_data = to_pyspark_vector(df, config)

            model = None
            if config["training_type"] == 'lr':
                logger.info("Starting Linear Regression training.")
                model = linear_regression(train_data, test_data, config["prediction_feature"], config)
            elif config["training_type"] == 'nn':
                logger.info("Starting Neural Network training.")
                model = neural_network_regression(train_data, test_data, config["prediction_feature"], config)

            if model:
                model_path = os.path.join(MODELS_DIR, args.out)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                model.save(model_path)
                logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
