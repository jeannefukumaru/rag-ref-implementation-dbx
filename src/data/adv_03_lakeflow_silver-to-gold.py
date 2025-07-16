# Databricks notebook source
"""
Lakeflow Data Pipeline: Silver to Gold Layer Enhancement

This module implements the final transformation stage of a data lakehouse pipeline that
enriches silver layer data with advanced processing to create analytics-ready gold layer
data. The pipeline focuses on multimodal document processing and includes:

1. Multi-modal text extraction from PDF files and images using PyPDF2
2. Document chunking using LangChain's RecursiveCharacterTextSplitter
3. AI-powered topic extraction using Databricks' LLM endpoints
4. Structured data preparation optimized for vector search and analytics

The gold layer represents the highest quality, most refined data that serves as the
foundation for business intelligence, machine learning models, and advanced analytics
workflows. This implementation specifically handles unstructured document processing
with LLM-enhanced metadata extraction.
"""

# MAGIC %pip install transformers==4.49.0 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch==0.55 mlflow[databricks] torch==2.3.0 markdownify==0.14.1 pypdf2 langchain

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("gold_table_name", "autoloader_etl_test_gold")
dbutils.widgets.text("silver_table_name", "autoloader_etl_test_silver")
dbutils.widgets.text("catalog_name", "users")
dbutils.widgets.text("schema_name", "jeanne_choo")


gold_table_name = dbutils.widgets.get("gold_table_name")
silver_table_name = dbutils.widgets.get("silver_table_name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

gold_table_name = f"{catalog_name}.{schema_name}.{gold_table_name}"
silver_table_name = f"{catalog_name}.{schema_name}.{silver_table_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Multi-modal Text Extraction

# COMMAND ----------

# DBTITLE 1,Extract text from PDF files
from io import BytesIO
from PyPDF2 import PdfReader
from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
import dlt

@pandas_udf("string")
def extract_pdf_text_udf(pdf_bytes_series: pd.Series) -> pd.Series:
    def extract(pdf_bytes):
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            return ""
    return pdf_bytes_series.apply(extract)

@dlt.table(
  comment="Extract text and topic from raw PDF content"
)
def text_extraction_silver():
  # Limit the number of rows to 2 as in the SQL version
  silver_df = spark.read.table(f"{silver_table_name}")
  # Inject the JSON schema variable into the ai_query call using an f-string.
  return silver_df.filter(F.col("path").endswith(".pdf")).withColumn(
    "extracted_text",
    extract_pdf_text_udf(F.col("content"))
)

# COMMAND ----------

# DBTITLE 1,Extract text from image files
# image_desc_sql_string = """SELECT *, ai_query("databricks-llama-4-maverick", "I am giving you some images. construct a description for the image",
#    responseFormat => 'STRUCT<extracted_text: STRUCT<description: STRING>>',
#    files => content
#  ) as extracted_text FROM READ_FILES("/Volumes/users/jeanne_choo/scb-dabs-volume/*.png")"""

# @dlt.table(
#   comment="Extract text description from raw content"
# )
# def image_description_silver():
#   return spark.sql(image_desc_sql_string)

# COMMAND ----------

# DBTITLE 1,Join image and PDF extraction tables
@dlt.table(
  comment="Combine text extracted and image extraction tables"
)
# def text_image_extraction_silver():
#   text_df = spark.table("text_extraction_silver")
#   image_df = spark.table("image_description_silver")
#   return text_df.unionByName(image_df, allowMissingColumns=True)
def text_image_extraction_silver():
  text_df = spark.table("text_extraction_silver")
  # image_df = spark.table("image_description_silver")
  return text_df

# COMMAND ----------

# MAGIC %md
# MAGIC #Metadata enrichment and chunking

# COMMAND ----------

# DBTITLE 1,Implement chunking UDF
from typing import Iterator
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Reduce the arrow batch size as our PDF can be big in memory (classic compute only)
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def chunk_sentences(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    #Sentence splitter to split on sentences
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,         # maximum number of characters in each chunk
    chunk_overlap=20,       # number of overlapping characters between chunks
    length_function=len)     # function to determine the "length" (defaults to length

    def split_text(text):
      return text_splitter.split_text(text)
    
    for x in batch_iter:
        yield x.apply(split_text)

# COMMAND ----------

# DBTITLE 1,Implement AI Functions for LLM Information Extraction
from pyspark.sql.functions import col, from_json, to_json, explode
import json


PROMPT = """You are an AI assistant specialized in analyzing documentation. 
Your task is to extract relevant information from a given technical document. 
Your output must be a structured JSON object.

Instructions:
1. Carefully read the entire  document provided at the end of this prompt.
2. Extract the relevant information.
3. Present your findings in JSON format as specified below.

Important Notes:
- Extract only relevant information. 
- Consider the context of the entire contract when determining relevance.
- Do not be verbose, only respond with the correct format and information.
- Some questions may have no relevant excerpts. Just return "N/A" or ["N/A"] depending on the expected type in this case.
- Do not include additional JSON keys beyond the ones listed here.
- Do not include the same key multiple times in the JSON.

Expected JSON keys and explanation of what they are:
- topic: the main topic addressed in the document"""

response_format = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "topic_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
            },
            "strict": True,
        },
    },
})

# define query
ai_query_expr = f"""
  ai_query(
    endpoint => 'databricks-llama-4-maverick',
    request => CONCAT('{PROMPT}', extracted_text),
    responseFormat => '{response_format}'
  ) AS topic
  """

# the json schema of the LLM response string which we want to unpack
json_schema = "STRUCT<topic STRING>"

@dlt.table(
  comment="extract topics from document text using LLM inference"
)
def autoloader_etl_test_gold():
  df = spark.read.table("text_image_extraction_silver")
  return df.selectExpr(
    "*", ai_query_expr
  ).withColumn("parsed_topic", from_json(col("topic"), json_schema)
  ).withColumn("parsed_topic_str", to_json(col("parsed_topic")) # convert to json compatible string for vector search
  ).withColumn("chunked_text", explode(chunk_sentences(col("extracted_text")))
  ).drop("_change_type", "_commit_version", "_commit_timestamp", "content", "parsed_topic")
