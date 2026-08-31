import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, from_json, from_unixtime, hour, month, year
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


def main():
    spark = (
        SparkSession.builder.appName("KafkaToBronzeStructuredStreaming")
        .config("spark.sql.catalogImplementation", "in-memory")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ===== Configuración =====
    bootstrap_servers = spark.conf.get("spark.kafka.bootstrap.servers", "kafka:9092")
    topic = spark.conf.get("spark.kafka.topic", "events")
    starting_offsets = spark.conf.get("spark.kafka.startingOffsets", "earliest")
    checkpoint_location = spark.conf.get(
        "spark.bronze.checkpoint.location", "s3a://bronze/checkpoints/eventos_batch"
    )
    output_path = spark.conf.get("spark.bronze.output.path", "s3a://bronze/eventos_batch")

    print(f"Reading from Kafka: {bootstrap_servers} (topic: {topic})")
    print(f"Checkpoint location: {checkpoint_location}")
    print(f"Output path: {output_path}")

    # ===== Esquema =====
    schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("product", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("timestamp", DoubleType(), True),  # epoch seconds
        ]
    )

    # ===== Kafka Structured Streaming Source =====
    # Nota: startingOffsets aplica solo en la primera ejecución si no existe checkpoint.
    # En ejecuciones posteriores, Spark lee automáticamente desde el último offset confirmado en el checkpoint.
    df_kafka = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    # ===== Parse JSON =====
    df = (
        df_kafka.selectExpr("CAST(value AS STRING) AS json_payload")
        .select(from_json(col("json_payload"), schema).alias("data"))
        .select("data.*")
    )

    # ===== Timestamp y Particiones =====
    df = df.withColumn("event_time", from_unixtime(col("timestamp")))

    df = (
        df.withColumn("year", year("event_time"))
        .withColumn("month", month("event_time"))
        .withColumn("day", dayofmonth("event_time"))
        .withColumn("hour", hour("event_time"))
    )

    # ===== Escritura con Checkpoint y Trigger AvailableNow =====
    # trigger(availableNow=True) procesa todos los datos disponibles en micro-batches y termina.
    # El checkpoint garantiza exactamente-una-vez (idempotencia en reintentos) y evita reprocesar el topic entero.
    query = (
        df.writeStream.format("parquet")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_location)
        .option("path", output_path)
        .partitionBy("year", "month", "day", "hour")
        .trigger(availableNow=True)
        .start()
    )

    query.awaitTermination()
    print("Bronze Structured Streaming batch ingestion completed successfully.")

    spark.stop()
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Spark job failed:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
