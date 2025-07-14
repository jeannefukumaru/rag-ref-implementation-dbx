# Databricks notebook source
# MAGIC %md
# MAGIC ## Ingest Source to Delta Lake
# MAGIC We ingest the raw binary pdf-files to the Bronze table.
# MAGIC This will allow us to keep track of upstream changes to the
# MAGIC documents, maintain history, roll-back and re-run downstream pipelines.

# COMMAND ----------

import pyspark.sql.functions as func

# COMMAND ----------

# DBTITLE 1,Define input parameters for jobs
dbutils.widgets.text("source_doc_folder", "")
source_folder = dbutils.widgets.get("source_doc_folder")

dbutils.widgets.text("catalog_name", "users",)
catalog_name = dbutils.widgets.get("catalog_name")

dbutils.widgets.text("schema_name", "jeanne_choo",)
schema_name = dbutils.widgets.get("schema_name")

dbutils.widgets.text("project_name", "jeannes_chatbot",)
project_name = dbutils.widgets.get("project_name")

dbutils.widgets.text("embedding_endpoint_name", "databricks-gte-large-en")
embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

dbutils.widgets.text("vector_search_endpoint_name", "one-env-shared-endpoint-0")
vs_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")

dbutils.widgets.text(
    "vector_search_index_name",
    "jeannes-chatbot_vs_index_direct",
)
vs_index_name = dbutils.widgets.get("vector_search_index_name")
vs_index_name = f"{catalog_name}.{schema_name}.{vs_index_name}"

# COMMAND ----------
bronze_table_name = f"{catalog_name}.{schema_name}.{project_name}_bronze"
silver_table_name = f"{catalog_name}.{schema_name}.{project_name}_silver"
gold_table_name = f"{catalog_name}.{schema_name}.{project_name}_gold"


# COMMAND ----------

# DBTITLE 1,Load raw sources to Bronze table (as binaries)
df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(source_folder)
    .withColumnRenamed("path", "input_file")
    .withColumn("_datetime", func.current_timestamp())
)

# COMMAND ----------

# DBTITLE 1,Save Bronze Table
df.write.mode("overwrite").saveAsTable(bronze_table_name)

# COMMAND ----------
import io
import pyspark.sql.functions as func
from pyspark.sql.types import MapType, StringType
from pypdf import PdfReader

def bronze_to_silver_pypdf(df_bronze):
    """
    Parse pdf documents simply through pypdf
    """
    @func.udf(returnType=MapType(StringType(), StringType()))
    def parse_pdf(pdf_content):
        pdf = io.BytesIO(pdf_content)
        reader = PdfReader(pdf)
        return {str(count): page.extract_text() for count, page in enumerate(reader.pages)}

    df_parsed = df_bronze.select(
        func.col("input_file"),
        parse_pdf("content").alias("parsed_pdf_pages"),
    )

    return (
        df_parsed.select("*", func.explode("parsed_pdf_pages"))
        .withColumnRenamed("key", "page_nr")
        .withColumnRenamed("value", "page_content")
        .drop("parsed_pdf_pages")
    )

# COMMAND ----------

# DBTITLE 1,Load Bronze
df = spark.read.table(bronze_table_name)
df_silver = bronze_to_silver_pypdf(df)

regular_expression = r".*\/([^\/]+)\/([^\/]+)\.pdf"

# some filename cleaning
df_silver = df_silver.withColumn(
    "document_name",
    func.concat(
        func.regexp_replace(
            func.lower(func.regexp_extract("input_file", regular_expression, 1)),
            " ",
            "_",
        ),
        func.lit("/"),
        func.concat(
            func.regexp_replace(
                func.lower(func.regexp_extract("input_file", regular_expression, 2)),
                " ",
                "_",
            )
        ),
    ),
).withColumn("_datetime", func.current_timestamp())

# DBTITLE 1,Write Silver table
df_silver.write.mode("overwrite").saveAsTable(silver_table_name)

# DBTITLE 1,Set tags for Silver table
tag_query = f"ALTER TABLE {silver_table_name} SET TAGS ('pdf_parser'='pdf_parser')"
spark.sql(tag_query)

# COMMAND ----------

# DBTITLE 1,Load Silver
df_silver = spark.read.table(silver_table_name).select(
    "input_file", "page_nr", "page_content", "document_name"
)

# COMMAND ----------


# DBTITLE 1,Spark's Vectorized (Pandas)-UDF for Langchain-based chunking
from typing import Iterator, Union, List, Dict
from pyspark.sql import Column
from pyspark.sql.types import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

def data_prep_udf(function, returnType):
    def _map_pandas_func(
        iterator: Iterator[Union[pd.Series, pd.DataFrame]]
    ) -> Iterator[pd.Series]:
        for x in iterator:
            if type(x) is pd.DataFrame:
                result = x.apply(lambda y: function(y), axis=1)
            else:
                result = x.map(lambda y: function(y))
            yield result

    return func.pandas_udf(f=_map_pandas_func, returnType=returnType)


def recursive_character_split(
    col: Column,
    chunk_size: int,
    chunk_overlap: int,
    explode=True,
):
    split_schema = ArrayType(
        StructType(
            [
                StructField("content_chunk", StringType(), False),
            ]
        )
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _split_char_recursive(content: str) -> List[Dict[str, Union[str, int]]]:
        chunks = text_splitter.split_text(content)
        return [
            {"content_chunk": doc}
            for doc in chunks
        ]

    split_func = data_prep_udf(_split_char_recursive, split_schema)(col)
    if explode:
        return func.explode(split_func)
    return split_func


# COMMAND ----------

# DBTITLE 1,Gold = Chunked Silver 

chunk_size = 475
chunk_overlap = 50

df_gold = (
    df_silver.withColumn(
        "page_chunks",
        recursive_character_split(
            func.col("page_content"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            explode=True,
        ),
    )
    .withColumn("content_chunk", func.col("page_chunks.content_chunk"))
    .withColumn("_datetime", func.current_timestamp())
    .drop("page_chunks")
)

# COMMAND ----------

# Note that we need to enable Change Data Feed on the table to create the index
spark.sql(
    f"""
CREATE TABLE IF NOT EXISTS {gold_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  input_file STRING,
  page_nr STRING,
  document_name STRING,
  content_chunk STRING,
  _datetime TIMESTAMP,
  page_content STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
          """
)


# DBTITLE 1,Write Gold
df_gold.write.mode("overwrite").saveAsTable(gold_table_name)

# COMMAND ----------

import datetime

import pyspark.sql.functions as func
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient



vs_index_name = dbutils.widgets.get("vector_search_index_name")

# COMMAND ----------

vsc = VectorSearchClient()
w = WorkspaceClient()

# COMMAND ----------
try:
  vs_endpoint = vsc.get_endpoint(vs_endpoint_name)
except Exception as e:
  print(f"Vectorsearch endpoint {vs_endpoint_name} is not available. Creating it ...")
  vs_endpoint = vsc.create_endpoint_and_wait(name=vs_endpoint_name, timeout=datetime.timedelta(seconds=3600))

try:
  index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_name)
  print(f"Syncing index {vs_index_name}")
  index.sync()
except Exception as e:
  print(f"The index {vs_index_name} does not exist. Creating it now ...")
  index = vsc.create_delta_sync_index_and_wait(
      endpoint_name=vs_endpoint_name,
      index_name=vs_index_name,
      primary_key="id",
      pipeline_type="TRIGGERED",
      source_table_name=gold_table_name,
      embedding_source_column="content_chunk",
      embedding_model_endpoint_name=embedding_endpoint_name,
  )
