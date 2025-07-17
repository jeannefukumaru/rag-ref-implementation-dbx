# Databricks notebook source
# MAGIC %md 
# MAGIC # Agent Evaluation Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC This Databricks notebook (`02_agent_evaluation.py`) provides examples for evaluating AI agents, specifically designed for RAG (Retrieval-Augmented Generation) chatbots. It demonstrates best practices for creating evaluation datasets, running assessments, and setting up continuous monitoring.
# MAGIC
# MAGIC ## Key Components
# MAGIC
# MAGIC ### 1. Synthetic Dataset Generation
# MAGIC - **Purpose**: Creates evaluation datasets from source documents to test agent performance
# MAGIC - **Benefits**: 
# MAGIC   - Fix known issues by adding problematic examples
# MAGIC   - Prevent regressions with a "golden set" of examples
# MAGIC   - Compare different versions (prompts, models, app logic)
# MAGIC   - Target specific features (safety, domain knowledge, edge cases)
# MAGIC
# MAGIC ### 2. Evaluation Workflow
# MAGIC - **Steps**:
# MAGIC   1. Generate synthetic evaluation data from your source documents.
# MAGIC   2. Save the generated evaluation dataset as an `EvaluationDataset` in Unity Catalog.
# MAGIC   3. Run evaluation with pre-defined scorers (ground truth required and not required).
# MAGIC   4. Deploy your scorers for continuous monitoring in production.
# MAGIC
# MAGIC #### Step 1: Generate Synthetic Evaluation Data
# MAGIC ```python
# MAGIC evals = generate_evals_df(
# MAGIC     docs,
# MAGIC     num_evals=10,
# MAGIC     agent_description=agent_description,
# MAGIC     question_guidelines=question_guidelines
# MAGIC )
# MAGIC ```
# MAGIC - Creates 10 synthetic evaluation examples
# MAGIC - Distributes evaluations across provided documents
# MAGIC - Uses agent description and question guidelines for context
# MAGIC
# MAGIC #### Step 2: Save Evaluation Dataset
# MAGIC - Saves evaluation datasets as `EvaluationDataset` in Unity Catalog
# MAGIC - Creates table: `{catalog_name}.{schema_name}.spark_docs_eval`
# MAGIC - Merges production traces from recent activity (last 10 minutes)
# MAGIC
# MAGIC #### Step 3: Predefined Scorers
# MAGIC The notebook uses MLflow's built-in evaluation scorers:
# MAGIC
# MAGIC **Ground Truth Required:**
# MAGIC - `Correctness()`: Measures factual accuracy
# MAGIC - `RetrievalSufficiency()`: Evaluates if retrieved context is sufficient
# MAGIC
# MAGIC **No Ground Truth Required:**
# MAGIC - `RelevanceToQuery()`: Assesses response relevance to user query
# MAGIC - `RetrievalGroundedness()`: Checks if response is grounded in retrieved context
# MAGIC - `RetrievalRelevance()`: Evaluates relevance of retrieved documents
# MAGIC - `Safety()`: Identifies harmful or inappropriate content
# MAGIC - `Guidelines()`: Custom scorer for specific requirements
# MAGIC
# MAGIC #### Step 4: Continuous Monitoring
# MAGIC Sets up external monitoring with:
# MAGIC - **Sampling Rate**: 100% of requests
# MAGIC - **Built-in Judges**: Safety, groundedness, relevance, chunk relevance
# MAGIC - **Custom Guidelines**: MLflow-specific responses, customer service tone
# MAGIC
# MAGIC ## Configuration Parameters
# MAGIC
# MAGIC ### Widget Configuration
# MAGIC - `catalog name`: Unity Catalog namespace (default: "users")
# MAGIC - `schema name`: Schema within catalog (default: "jeanne_choo")
# MAGIC
# MAGIC ### Model Endpoints
# MAGIC - Primary model: `databricks-meta-llama-3-1-405b-instruct`
# MAGIC - Agent endpoint: `agents_users-jeanne_choo-jeanne_chatbot`
# MAGIC
# MAGIC ## Usage Notes
# MAGIC
# MAGIC ### Best Practices
# MAGIC 1. **Dataset Coverage**: Ensure evaluation covers all document types
# MAGIC 2. **Trace Collection**: Monitor recent production activity for real-world examples
# MAGIC 3. **Scorer Selection**: Choose appropriate scorers based on ground truth availability
# MAGIC 4. **Continuous Monitoring**: Set up automated quality assessments for production
# MAGIC
# MAGIC ### Customization Options
# MAGIC - Modify `agent_description` for different use cases
# MAGIC - Adjust `question_guidelines` for specific user personas
# MAGIC - Add custom scorers with `GuidelinesJudge` for domain-specific requirements
# MAGIC - Configure sampling rates for different assessment types
# MAGIC
# MAGIC ## Output
# MAGIC The notebook generates:
# MAGIC - Synthetic evaluation datasets displayed in tabular format
# MAGIC - MLflow evaluation results with detailed metrics
# MAGIC - Unity Catalog tables for persistent storage
# MAGIC - External monitoring configuration for production use
# MAGIC
# MAGIC This framework enables systematic evaluation of AI agents with both synthetic and production data, ensuring consistent quality and performance monitoring.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow>=3.1.1 databricks-langchain uv langgraph==0.3.4 langgraph-checkpoint-postgres psycopg pytest databricks-agents

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog name", "users")
dbutils.widgets.text("schema name", "jeanne_choo")
catalog_name = dbutils.widgets.get("catalog name")
schema_name = dbutils.widgets.get("schema name")

# COMMAND ----------

# MAGIC %md 
# MAGIC # 1. Create evaluation datasets based on source documents
# MAGIC Evaluation datasets help you:
# MAGIC
# MAGIC - Fix known issues: Add problematic examples from production to repeatedly test fixes
# MAGIC - Prevent regressions: Create a "golden set" of examples that must always work correctly
# MAGIC - Compare versions: Test different prompts, models, or app logic against the same data
# MAGIC - Target specific features: Build specialized datasets for safety, domain knowledge, or edge cases

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Create synthetic datasets 

# COMMAND ----------

import mlflow
from databricks.agents.evals import generate_evals_df
import pandas as pd

# These documents can be a Pandas DataFrame or a Spark DataFrame with two columns: 'content' and 'doc_uri'.
docs = pd.DataFrame.from_records(
    [
      {
        'content': f"""
            Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java,
            Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set
            of higher-level tools including Spark SQL for SQL and structured data processing, pandas API on Spark for pandas
            workloads, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for incremental
            computation and stream processing.
        """,
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/'
      },
      {
        'content': f"""
            Spark’s primary abstraction is a distributed collection of items called a Dataset. Datasets can be created from Hadoop InputFormats (such as HDFS files) or by transforming other Datasets. Due to Python’s dynamic nature, we don’t need the Dataset to be strongly-typed in Python. As a result, all Datasets in Python are Dataset[Row], and we call it DataFrame to be consistent with the data frame concept in Pandas and R.""",
        'doc_uri': 'https://spark.apache.org/docs/3.5.2/quick-start.html'
      }
    ]
)

agent_description = """
The Agent is a RAG chatbot that answers questions about using Spark on Databricks. The Agent has access to a corpus of Databricks documents, and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. The corpus covers a lot of info, but the Agent is specifically designed to interact with Databricks users who have questions about Spark. So questions outside of this scope are considered irrelevant.
"""
question_guidelines = """
# User personas
- A developer who is new to the Databricks platform
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- what API lets me parallelize operations over rows of a delta table?
- Which cluster settings will give me the best performance when using Spark?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 10

evals = generate_evals_df(
    docs,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, some documents will not have any evaluations generated. 
    # For details about how `num_evals` is used to distribute evaluations across the documents, 
    # see the documentation: 
    # AWS: https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html#num-evals. 
    # Azure: https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/synthesize-evaluation-set 
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    agent_description=agent_description,
    question_guidelines=question_guidelines
)

display(evals)

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results. You can use this as a baseline for your agent.
results = mlflow.evaluate(
  model="endpoints:/databricks-meta-llama-3-1-405b-instruct",
  data=evals,
  model_type="databricks-agent"
)

# Note: To use a different model serving endpoint, use the following snippet to define an agent_fn. Then, specify that function using the `model` argument.
# MODEL_SERVING_ENDPOINT_NAME = '...'
# def agent_fn(input):
#   client = mlflow.deployments.get_deploy_client("databricks")
#   return client.predict(endpoint=MODEL_SERVING_ENDPOINT_NAME, inputs=input)

# COMMAND ----------

results.tables["eval_results"]

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Save generated evaluation dataset as a `EvaluationDataset` in Unity Catalog

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
import time
from databricks.connect import DatabricksSession

# 1. Create an evaluation dataset

# Replace with a Unity Catalog schema where you have CREATE TABLE permission
uc_schema = f"{catalog_name}.{schema_name}"
# This table will be created in the above UC schema
evaluation_dataset_table_name = "spark_docs_eval"

eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}",
)
print(f"Created evaluation dataset: {uc_schema}.{evaluation_dataset_table_name}")

# 2. Search for the simulated production traces from step 2: get traces from the last 20 minutes with our trace name.
ten_minutes_ago = int((time.time() - 10 * 60) * 1000)

traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {ten_minutes_ago} AND "
                 f"attributes.status = 'OK'",
                  order_by=["attributes.timestamp_ms DESC"]
)

print(f"Found {len(traces)} successful traces from beta test")

# 3. Add the traces to the evaluation dataset
eval_dataset.merge_records(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# COMMAND ----------

eval_dataset_spark = spark.table(eval_dataset.name)
display(eval_dataset_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Run evaluation with pre-defined scorers

# COMMAND ----------

predict_fn = mlflow.genai.to_predict_fn("endpoints:/agents_users-jeanne_choo-jeanne_chatbot")

# COMMAND ----------

from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
)


# Run predefined scorers that require ground truth
mlflow.genai.evaluate(
    data=eval_dataset_spark,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        # RelevanceToQuery(),
        # RetrievalGroundedness(),
        # RetrievalRelevance(),
        RetrievalSufficiency(),
        # Safety(),
    ],
)

# COMMAND ----------

# Run predefined scorers that do NOT require ground truth
mlflow.genai.evaluate(
    data=eval_dataset_spark,
    predict_fn=predict_fn,
    scorers=[
        # Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        # RetrievalSufficiency(),
        Safety(),
        Guidelines(name="does_not_mention", guidelines="The response not mention the fact that provided context exists.")
    ],
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # 4. Deploy your scorers for continuous monitoring

# COMMAND ----------

# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import create_external_monitor, AssessmentsSuiteConfig, BuiltinJudge, GuidelinesJudge

external_monitor = create_external_monitor(
    # Change to a Unity Catalog schema where you have CREATE TABLE permissions.
    catalog_name=f"{catalog_name}",
    schema_name=f"{schema_name}",
    assessments_config=AssessmentsSuiteConfig(
        sample=1.0,  # sampling rate
        assessments=[
            # Predefined scorers "safety", "groundedness", "relevance_to_query", "chunk_relevance"
            BuiltinJudge(name="safety"),  # or {'name': 'safety'}
            BuiltinJudge(
                name="groundedness", sample_rate=0.4
            ),  # or {'name': 'groundedness', 'sample_rate': 0.4}
            BuiltinJudge(
                name="relevance_to_query"
            ),  # or {'name': 'relevance_to_query'}
            BuiltinJudge(name="chunk_relevance"),  # or {'name': 'chunk_relevance'}
            # Guidelines can refer to the request and response.
            GuidelinesJudge(
                guidelines={
                    # You can have any number of guidelines, each defined as a key-value pair.
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],  # Must be an array of strings
                    "customer_service_tone": [
                        """The response must maintain our brand voice which is:
    - Professional yet warm and conversational (avoid corporate jargon)
    - Empathetic, acknowledging emotional context before jumping to solutions
    - Proactive in offering help without being pushy

    Specifically:
    - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
    - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
    - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
    - The response must end with a specific next step or open-ended offer to help, not generic closings"""
                    ],
                }
            ),
        ],
    ),
)

print(external_monitor)

# COMMAND ----------

external_monitor

# COMMAND ----------

