#!./venv/bin/python

import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, first, lit
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import os


CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streaming/configs")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def valid_config(name: str) -> str:
    if name not in os.listdir(CONFIG_DIR):
        raise argparse.ArgumentTypeError(f"Error: The provided streaming config does not exist")
    return name

def valid_model(name: str) -> str:
    if name not in os.listdir(MODELS_DIR):
        raise argparse.ArgumentTypeError(f"Error: The provided prediction model does not exist")
    return name

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for collecting and processing streaming data.")
    parser.add_argument('-c', '--config', 
                        type=valid_config, 
                        required=True, 
                        choices=[config for config in os.listdir(CONFIG_DIR)], 
                        help='Filename of the config to use')
    parser.add_argument('-m', '--model', 
                        type=valid_model, 
                        required=True, 
                        choices=[model_name for model_name in os.listdir(MODELS_DIR)], 
                        help='Name of the prediction model to use')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with open(os.path.join(CONFIG_DIR, args.config)) as configfile:
        config = json.load(configfile)

        spark: SparkSession = SparkSession.builder \
            .appName("KafkaSparkStreaming") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        model: LinearRegressionModel = LinearRegressionModel.load(os.path.join(MODELS_DIR, args.model))

        message_schema = StructType() \
            .add("id", StringType()) \
            .add("id_macchina", StringType()) \
            .add("valore", DoubleType()) \
            .add("data_registrazione", TimestampType()) \
            .add("createdAt", TimestampType()) \
            .add("updatedAt", TimestampType()) \
            .add("nome_parametro", StringType()) \
            .add("tipo_dato", StringType())

        kafka_df: DataFrame = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", f"{config['broker_address']}:{config['broker_port']}") \
            .option("subscribe", config["topic"]) \
            .option("startingOffsets", "latest") \
            .load()

        parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_string") \
            .select(from_json(col("json_string"), message_schema).alias("data")) \
            .select("data.*") \
            .drop(config["prediction_feature"]) \
            .select("data_registrazione", "nome_parametro", "valore")  

        parsed_df = parsed_df.withColumn("constant_key", lit(1))

        # Data pivoting without using pivot() method (not allowed in streaming mode)
        pivoted_df = parsed_df.groupBy("constant_key").agg(
            *[first(when(col("nome_parametro") == parametro, col("valore")), ignorenulls=True).alias(parametro) for parametro in config["params"]]
        ).drop("constant_key")

        assembler = VectorAssembler(inputCols=config["params"], outputCol="features", handleInvalid="keep")
        vector_df = assembler.transform(pivoted_df)

        predictions_df = model.transform(vector_df)

        query = predictions_df.writeStream \
            .outputMode("update") \
            .format("console") \
            .trigger(processingTime="3 seconds") \
            .start()

        query.awaitTermination()
