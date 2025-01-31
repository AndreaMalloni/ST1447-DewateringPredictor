from datetime import datetime
import json
import logging
import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, row_number, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'{os.path.abspath("ST1447-DewateringPredictor/data/logs")}/{datetime.now()}.log',
    filemode='a'  
)
logger = logging.getLogger(__name__)

def log(config, level, message):
    if config["logging"]:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)

def process_data(df: DataFrame, configName: str) -> tuple:
    try:
        configPath = os.path.join(os.path.abspath("ST1447-DewateringPredictor/data/configs"), configName)

        with open(configPath) as configfile:
            config = json.load(configfile)

            log(config, "info", "Inizio elaborazione dei dati.")
            log(config, "info", f"Caricamento configurazione da 'configs/{config}'.")

            if not config["all"]:
                log(config, "info", "Filtraggio dei dati in base ai parametri specificati nella configurazione.")
                df = df.filter(col("nome_parametro").isin(config["params"]))

            log(config, "info", "Creazione di WindowSpec per ordinare i dati per 'data_registrazione'.")
            windowSpec = Window.partitionBy("nome_parametro").orderBy("data_registrazione")
            df = df.withColumn("row_id", row_number().over(windowSpec))

            log(config, "info", "Pivot dei dati e aggregazione per 'nome_parametro'.")
            df = df.select("row_id", "nome_parametro", "valore") \
                .groupBy("row_id") \
                .pivot("nome_parametro") \
                .agg({"valore": "first"})
            df = df.drop("row_id")

            log(config, "info", "Raggruppamento dei dati per identificare duplicati con count > 1.")
            df = df.groupBy(df.columns).count().filter("count > 1")
            df = df.dropDuplicates()

            log(config, "info", "Schema del DataFrame risultante:")
            if config["logging"]: df.printSchema()

            for column in df.columns:
                log(config, "info", f"Analisi della colonna '{column}'...")
                max_val = df.agg({column: "max"}).collect()[0][0]
                min_val = df.select(
                    when(col(column) != 0, col(column)).alias(column)
                ).agg({column: "min"}).collect()[0][0]
                row_count = df.filter(col(column).isNotNull()).count()

                log(config, "info", f"Valore massimo per {column}: {max_val}")
                log(config, "info", f"Valore minimo per {column} (escludendo 0): {min_val}")
                log(config, "info", f"Numero di righe non nulle per {column}: {row_count}\n")

            log(config, "info", "Creazione del VectorAssembler per la trasformazione dei dati.")
            assembler = VectorAssembler(
                inputCols=config["params"] if config["params"] else df.columns,
                outputCol="features"
            )
            data = assembler.transform(df)

            log(config, "info", "Selezione delle colonne 'features' e 'PV16_Torbidit\u00e0Chiarificato'.")
            final_data = data.select("features", "PV16_Torbidit\u00e0Chiarificato")

            log(config, "info", "Suddivisione dei dati in train e test set.")
            train_data, test_data = final_data.randomSplit(
                [config["partitions"]["train"], config["partitions"]["test"]], 
                seed=config["partitions"]["seed"]
            )

            log(config, "info", "Elaborazione completata con successo.")
            return train_data, test_data

    except Exception as e:
        log(config, "error", f"Errore durante l'elaborazione dei dati: {e}")
        raise


if __name__ == "__main__":
    pass