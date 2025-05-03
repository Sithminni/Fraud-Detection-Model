from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Initialize Spark session
spark = SparkSession.builder.appName("FraudBatchModel").getOrCreate()

# Load dataset
df = spark.read.csv("data/fraud_detection.csv", header=True, inferSchema=True)

# Encode categorical variable
indexer = StringIndexer(inputCol="type", outputCol="type_indexed")
df = indexer.fit(df).transform(df)

# Assemble features
assembler = VectorAssembler(
    inputCols=["type_indexed", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
    outputCol="features"
)
df = assembler.transform(df)

# Prepare final DataFrame
data = df.select("features", df["isFraud"].alias("label"))

# Train/test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
model = rf.fit(train_data)

# Save model
model.save("models/rf_model")

print("✅ Model trained and saved.")
