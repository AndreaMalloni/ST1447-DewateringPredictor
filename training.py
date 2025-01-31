#!/home/andrea/TBDM/ST1447-DewateringPredictor/venv/bin/python

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from data.data import process_data
import numpy as np
import tensorflow as tf

def train_model(train_data, test_data):
    lr = LinearRegression(featuresCol="features", labelCol="PV16_TorbiditàChiarificato", predictionCol="Predicted_PV16_TorbiditàChiarificato", regParam=0.01)
    lr_model = lr.fit(train_data)

    predictions = lr_model.transform(test_data)

    evaluator = RegressionEvaluator(labelCol="PV16_TorbiditàChiarificato", predictionCol="Predicted_PV16_TorbiditàChiarificato", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data: {:.3f}".format(rmse))

    evaluator_r2 = RegressionEvaluator(labelCol="PV16_TorbiditàChiarificato", predictionCol="Predicted_PV16_TorbiditàChiarificato", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)
    print("R-squared (R2) on test data: {:.3f}".format(r2))


def to_numpy(df, features_col, label_col):
    features = np.array(df.select(features_col).rdd.map(lambda row: row[0]).collect())
    labels = np.array(df.select(label_col).rdd.map(lambda row: row[0]).collect())
    return features, labels

def train_model_nn(train_data, test_data):
    train_features, train_labels = to_numpy(train_data, "features", "PV16_TorbiditàChiarificato")
    test_features, test_labels = to_numpy(test_data, "features", "PV16_TorbiditàChiarificato")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',  # Mean Squared Error
                  metrics=['mae'])  # Mean Absolute Error

    model.fit(train_features, train_labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    test_loss, test_mae = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.3f}")
    print(f"Test MAE: {test_mae:.3f}")

    predictions = model.predict(test_features)

    ss_total = np.sum((test_labels - np.mean(test_labels)) ** 2)
    ss_residual = np.sum((test_labels - predictions.flatten()) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print(f"R-squared (R2) on test data: {r2:.3f}")


if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName("Linear Regression with PySpark MLlib") \
    .getOrCreate()

    df = spark.read.option("encoding", "ISO-8859-1").csv("ST1447-DewateringPredictor/data/resources/dati_macchina_01.csv", header=True, inferSchema=True)
    train, test = process_data(df, "standard.json")
    #df.show(100)



    #train_model_nn(train_data, test_data)

    #train_model(train_data, test_data)