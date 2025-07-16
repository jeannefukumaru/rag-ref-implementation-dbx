# Databricks notebook source
"""
Lakeflow Data Pipeline: Bronze to Silver Layer Transformation

This module implements the second stage of a data lakehouse pipeline that transforms
bronze layer data into a cleaner, more structured silver layer using Databricks
Delta Live Tables (DLT). The pipeline:

1. Creates a streaming table that reads Change Data Feed (CDF) from the bronze layer
2. Applies Change Data Capture (CDC) transformations to materialize clean data
3. Implements SCD Type 2 (Slowly Changing Dimension) to track historical changes
4. Maintains data lineage and supports incremental processing patterns

The silver layer serves as the cleaned, validated, and deduplicated data layer
that forms the foundation for downstream analytics and machine learning workflows.
"""

# MAGIC %md 
# MAGIC # Setup bronze CDF table as a streaming table

# COMMAND ----------

dbutils.widgets.text("bronze_table_name", "autoloader_etl_test_bronze")
dbutils.widgets.text("silver_table_name", "autoloader_etl_test_silver")
dbutils.widgets.text("catalog_name", "users")
dbutils.widgets.text("schema_name", "jeanne_choo")

bronze_table_name = dbutils.widgets.get("bronze_table_name")
silver_table_name = dbutils.widgets.get("silver_table_name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

bronze_table_name = f"{catalog_name}.{schema_name}.{bronze_table_name}"
silver_table_name = f"{catalog_name}.{schema_name}.{silver_table_name}"

# COMMAND ----------

from dlt import *
from pyspark.sql.functions import *

# Create the target bronze table
dlt.create_streaming_table("cdc_bronze", comment="New data incrementally ingested from bronze landing zone")

# Create an Append Flow to ingest the raw data into the bronze cdc table
@append_flow(
  target = "cdc_bronze",
  name = "bronze_cdc_flow"
)
def bronze_cdc_flow():
  return (spark.readStream
              .option("readChangeFeed", "true")
              .table(bronze_table_name))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Apply CDC feed to silver table

# COMMAND ----------

dlt.create_streaming_table(name=f"{silver_table_name}", comment="Clean, materialized unstructured documents")

dlt.create_auto_cdc_flow(
  target=f"{silver_table_name}",  # The customer table being materialized
  source="cdc_bronze",  # the incoming CDC
  keys=["path"],  # what we'll be using to match the rows to upsert
  sequence_by=col("modificationTime"),  # de-duplicate by operation date, getting the most recent value
  ignore_null_updates=False,
  apply_as_deletes=expr("_change_type = 'DELETE'"),  # DELETE condition
  # except_column_list=["operation", "operation_date", "_rescued_data"],
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Setup SCD Type 2 for tracking changes

# COMMAND ----------

# create the table
dlt.create_streaming_table(
    name="docs_history", comment="Slowly Changing Dimension Type 2 for docs"
)

# store all changes as SCD2
dlt.create_auto_cdc_flow(
    target="docs_history",
    source=f"{silver_table_name}",
    keys=["path"],
    sequence_by=col("modificationTime"),
    ignore_null_updates=False,
    apply_as_deletes=expr("_change_type = 'DELETE'"),
    stored_as_scd_type="2",
)  # Enable SCD2 and store individual updates