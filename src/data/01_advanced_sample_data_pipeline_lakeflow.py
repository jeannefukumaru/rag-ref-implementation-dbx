# Databricks notebook source
# MAGIC %pip install --quiet -U transformers==4.49.0 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch==0.55 mlflow[databricks] tiktoken==0.7.0 torch==2.3.0 llama-index==0.10.43 markdownify==0.14.1 pypdf2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Databricks notebook source
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *
import json

# COMMAND ----------

# MAGIC %md
# MAGIC # PDF Processing Pipeline with Lakeflow Declarative Pipelines
# MAGIC 
# MAGIC This pipeline demonstrates:
# MAGIC - Ingesting PDF files using Auto Loader with streaming tables
# MAGIC - Processing PDFs with unstructured library
# MAGIC - Creating embeddings for vector search
# MAGIC - Using declarative transformations for data quality

# COMMAND ----------

catalog = "users"
db = "jeanne_choo"

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{db}/volume_databricks_documentation"
path_to_pdfs = 'dbfs:'+volume_folder+"/databricks-pdf"
path_to_pdfs

# COMMAND ----------

# COMMAND ----------

# Raw PDF ingestion streaming table
@dlt.table(
    comment="Raw PDF files ingested from cloud storage",
    table_properties={
        "quality": "bronze"
    }
)
def raw_pdf_files():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", "/tmp/pdf_schema_checkpoint")
        .load(path_to_pdfs)
        .select(
            F.col("path").alias("file_path"),
            F.col("content").alias("pdf_content"),
            F.col("modificationTime").alias("file_modified_time"),
            F.current_timestamp().alias("ingestion_time")
        )
    )

# COMMAND ----------

# COMMAND ----------

# PDF text extraction streaming table
@dlt.table(
    comment="Extracted text content from PDF files",
    table_properties={
        "quality": "silver"
    }
)
@dlt.expect_or_drop("valid_text_content", "length(extracted_text) > 0")
def extracted_pdf_text():
    from PyPDF2 import PdfReader
    import io

    def extract_text_from_pdf(pdf_content):
        try:
            # Convert binary content to file-like object
            pdf_file = io.BytesIO(pdf_content)
            # Use PyPDF2 to extract text
            reader = PdfReader(pdf_file)
            text_content = ""
            
            # Extract text from all pages
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    
    # Register UDF
    extract_text_udf = F.udf(extract_text_from_pdf, StringType())
    
    return (
        dlt.read_stream("raw_pdf_files")
        .select(
            F.col("file_path"),
            extract_text_udf(F.col("pdf_content")).alias("extracted_text"),
            F.col("file_modified_time"),
            F.col("ingestion_time"),
            F.current_timestamp().alias("processing_time")
        )
    )

# COMMAND ----------

# COMMAND ----------

# Text chunking streaming table
@dlt.table(
    comment="Chunked text segments for embedding generation",
    table_properties={
        "quality": "silver"
    }
)
@dlt.expect_or_drop("valid_chunk", "length(chunk_text) > 50")
def chunked_text():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    def chunk_text(text, chunk_size=1000, chunk_overlap=200):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            return [f"Error chunking text: {str(e)}"]
    
    # Register UDF that returns array of strings
    chunk_text_udf = F.udf(chunk_text, ArrayType(StringType()))
    
    return (
        dlt.read_stream("extracted_pdf_text")
        .select(
            F.col("file_path"),
            F.col("extracted_text"),
            chunk_text_udf(F.col("extracted_text")).alias("chunks"),
            F.col("file_modified_time"),
            F.col("processing_time")
        )
        .select(
            F.col("file_path"),
            F.col("extracted_text"),
            F.posexplode(F.col("chunks")).alias("chunk_id", "chunk_text"),
            F.col("file_modified_time"),
            F.col("processing_time")
        )
        .select(
            F.concat(F.col("file_path"), F.lit("_"), F.col("chunk_id")).alias("chunk_id"),
            F.col("file_path"),
            F.col("chunk_text"),
            F.col("chunk_id").alias("chunk_index"),
            F.col("file_modified_time"),
            F.col("processing_time")
        )
    )
0

# COMMAND ----------

# COMMAND ----------

# Embeddings generation materialized view (batch processing for efficiency)
@dlt.table(
    comment="Text chunks with computed embeddings for vector search",
    table_properties={
        "quality": "gold"
    }
)
@dlt.expect_or_drop("valid_embedding", "size(embedding) > 0")
def text_embeddings():
    # This would typically use a pre-trained model or API
    def generate_embeddings(text):
        try:
            # Placeholder for actual embedding generation
            # In practice, you'd use sentence-transformers, OpenAI API, etc.
            import hashlib
            # Simple hash-based mock embedding for demonstration
            hash_obj = hashlib.md5(text.encode())
            # Convert to list of floats (mock 384-dimensional embedding)
            embedding = [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 
                        for i in range(0, min(len(hash_obj.hexdigest()), 32), 2)]
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.append(0.0)
            return embedding[:384]
        except Exception as e:
            return [0.0] * 384  # Return zero vector on error
    
    # Register UDF
    generate_embeddings_udf = F.udf(generate_embeddings, ArrayType(FloatType()))
    
    return (
        dlt.read("chunked_text")
        .select(
            F.col("chunk_id"),
            F.col("file_path"),
            F.col("chunk_text"),
            F.col("chunk_index"),
            generate_embeddings_udf(F.col("chunk_text")).alias("embedding"),
            F.col("file_modified_time"),
            F.col("processing_time"),
            F.current_timestamp().alias("embedding_generated_time")
        )
    )

# COMMAND ----------

# COMMAND ----------

# Final vector search ready table
@dlt.table(
    comment="Final table optimized for vector search operations",
    table_properties={
        "quality": "gold",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }
)
def vector_search_ready():
    return (
        dlt.read("text_embeddings")
        .select(
            F.col("chunk_id").alias("id"),
            F.col("file_path").alias("source_file"),
            F.col("chunk_text").alias("content"),
            F.col("embedding"),
            F.struct(
                F.col("chunk_index"),
                F.col("file_modified_time"),
                F.col("processing_time"),
                F.col("embedding_generated_time")
            ).alias("metadata")
        )
    )

# COMMAND ----------

