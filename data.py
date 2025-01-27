
from pyspark.sql.functions import col, row_number, when
from pyspark.sql.window import Window

DESIRED_PARAMETERS = [
    "PV02_GiriMinutoCoclea",
    "PV01_GiriMinutoTamburo",
    "PV06_MisuratorePortataPoly",
    "PV05_MisuratorePortataFanghi",
    "PV35_ConcentrazionePoly",
    "PV16_TorbiditàChiarificato"
    ]


def transform_data(df):
    df.printSchema()
    # df = df.filter(col("nome_parametro").isin(DESIRED_PARAMETERS))

    windowSpec = Window.partitionBy("nome_parametro").orderBy("data_registrazione")
    df = df.withColumn("row_id", row_number().over(windowSpec))

    df = df.select("row_id", "nome_parametro", "valore") \
        .groupBy("row_id") \
        .pivot("nome_parametro") \
        .agg({"valore": "first"})
    df = df.drop("row_id")
    df = df.groupBy(df.columns).count().filter("count > 1")
    df.dropDuplicates()

    for column in df.columns:
        max_val = df.agg({column: "max"}).collect()[0][0]
        min_val = df.select(
        when(col(column) != 0, col(column)).alias(column)
            ).agg({column: "min"}).collect()[0][0]
        row_count = df.filter(col(column).isNotNull()).count()

        print(f"Max value for {column}: {max_val}")
        print(f"Min value for {column} (excluding 0): {min_val}")
        print(f"Number of rows for {column}: {row_count}\n")
    

    return df