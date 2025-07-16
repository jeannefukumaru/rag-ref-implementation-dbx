# Databricks notebook source
"""
Lakeflow Data Ingestion Pipeline: Bronze Layer

This module implements the first stage of a data lakehouse pipeline that ingests 
raw data files into a bronze Delta table using Databricks Auto Loader. The pipeline:

1. Streams binary files from a specified volume path using cloudFiles format
2. Adds processing timestamp metadata to each file
3. Writes the stream to a bronze Delta table with checkpointing
4. Configures table constraints and enables Change Data Feed for downstream processing

The bronze layer serves as the landing zone for raw, unprocessed data files
in the lakehouse architecture, preserving data lineage and supporting incremental
processing patterns.
"""

# COMMAND ----------

from pyspark.sql import functions as F
dbutils.widgets.text("file_path", "/Volumes/users/jeanne_choo/scb-dabs-volume")
dbutils.widgets.text("bronze_table_name", "autoloader_etl_test_bronze")
dbutils.widgets.text("silver_table_name", "autoloader_etl_test_silver")
dbutils.widgets.text("username", "jeanne.choo@databricks.com")
dbutils.widgets.text("catalog_name", "users")
dbutils.widgets.text("schema_name", "jeanne_choo")


file_path = dbutils.widgets.get("file_path")
bronze_table_name = dbutils.widgets.get("bronze_table_name")
silver_table_name = dbutils.widgets.get("silver_table_name")
username = spark.sql("SELECT current_user()").first()[0]
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
checkpoint_path = f"/tmp/{username}/_checkpoint"

bronze_table_name = f"{catalog_name}.{schema_name}.{bronze_table_name}"
silver_table_name = f"{catalog_name}.{schema_name}.{silver_table_name}"

sdf = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", checkpoint_path)
    .option("allowoverwrites", "true")
    .load(file_path)
    # Add source file and processing time columns
    .select("*", F.current_timestamp().alias("processing_time"))
)

# Write stream to Delta table using availableNow trigger
(
    sdf.writeStream
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .toTable(bronze_table_name)
)
bronze_df = spark.table(bronze_table_name)
display(bronze_df)

# COMMAND ----------
spark.sql(f"""
ALTER TABLE {bronze_table_name}
ALTER COLUMN path SET NOT NULL""")

# COMMAND ----------
# spark.sql(f"""ALTER TABLE {bronze_table_name}
# ADD CONSTRAINT IF NOT EXISTS path_primary_key PRIMARY KEY (path)""")

# COMMAND ----------
# Set Delta table property to enable Change Data Feed
spark.sql(
    f"""
    ALTER TABLE {bronze_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

# COMMAND ----------

# cdf = (
#   spark.readStream
#     .option("readChangeFeed", "true")
#     .table(f"{bronze_table_name}")
# )
# display(cdf)

# COMMAND ----------

