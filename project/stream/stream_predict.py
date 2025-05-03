from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

from pyspark.ml.feature import StringIndexerModel, VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from influxdb_client import InfluxDBClient, Point, WriteOptions
import pandas as pd


# 1. Create Spark session
spark = SparkSession.builder \
    .appName("FraudStreamPrediction") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Define schema for streaming data
schema = StructType([
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True),
])

# 3. Read stream from Kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fraud_stream") \
    .load()

# 4. Parse JSON string
df = df.selectExpr("CAST(value AS STRING) as json")
df = df.select(from_json(col("json"), schema).alias("data")).select("data.*")

# 5. Convert "type" to numeric using StringIndexerModel from trained model (or inline)
indexer_model = StringIndexerModel.from_labels(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], inputCol="type", outputCol="type_indexed")
df = indexer_model.transform(df)

# 6. Assemble features
assembler = VectorAssembler(
    inputCols=["type_indexed", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
    outputCol="features"
)
df = assembler.transform(df)

# 7. Load the trained model
model = RandomForestClassificationModel.load("models/rf_model")

# 8. Apply model
predictions = model.transform(df)

# 9. Display output (predicted frauds)
query = predictions.select("amount", "type", "prediction") \
    .writeStream.outputMode("append").format("console").start()

influx_bucket = "fraud_bucket"
influx_org = "fraud_org"
influx_token = "_h-eCvP2VuNdGmAbAVGVqbX4nJOFqgDiUTr5BCfy15G0vHpW18mAMkCmUiRwGv9aG56rn2AYOfvKKh5zyJnQeg=="
influx_url = "http://localhost:8086"

client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

def write_to_influx(batch_df, batch_id):
    pdf = batch_df.select("amount", "type", "prediction").toPandas()
    for _, row in pdf.iterrows():
        print("Writing row to InfluxDB:", row)
        point = Point("fraud_predictions") \
            .tag("type", row["type"]) \
            .field("amount", float(row["amount"])) \
            .field("prediction", int(row["prediction"])) \
            .time(pd.Timestamp.now())
        write_api.write(bucket=influx_bucket, org=influx_org, record=point)


# Replace previous writeStream
predictions.writeStream.foreachBatch(write_to_influx).start().awaitTermination()

