from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType
from data import transform_data

spark = SparkSession.builder \
    .appName("KafkaSparkStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

kafka_broker = "localhost:9092"
topic = "dati_macchina"

schema = StructType() \
    .add("id", StringType()) \
    .add("id_macchina", StringType()) \
    .add("valore", DoubleType()) \
    .add("data_registrazione", TimestampType()) \
    .add("createdAt", TimestampType()) \
    .add("updatedAt", TimestampType()) \
    .add("nome_parametro", StringType()) \
    .add("tipo_dato", StringType())

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("subscribe", topic) \
    .load()

messages_df = kafka_df.selectExpr("CAST(value AS STRING)")
parsed_df = messages_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

def process_batch(batch_df, batch_id):
    #print(f"Processing batch {batch_id}")
    #data = transform_data(batch_df)
    batch_df.show(100)

query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
    .start()

# Mantiene il job in esecuzione continua
query.awaitTermination()
