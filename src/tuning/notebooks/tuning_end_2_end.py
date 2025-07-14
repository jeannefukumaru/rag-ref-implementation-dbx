# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
import json
import mlflow
from mlflow.utils.databricks_utils import get_notebook_path
from pathlib import Path

w = WorkspaceClient()

# COMMAND ----------
# DBTITLE 1, Define some required input parameters in notebook widgets

# provide the cluster id of the interactive cluster to use for the tuning
dbutils.widgets.text("cluster_id", "", label="Interactive Cluster ID")
cluster_id = dbutils.widgets.get("cluster_id")

# provide name of an existing mlflow experiment.
dbutils.widgets.text(
    "experiment_name",
    "",
    label="Name of existing Mlflow experiment",
)
experiment_name = dbutils.widgets.get("experiment_name")

# provide a name for the parent run shown in flow experiment (Optional)
dbutils.widgets.text("run_name", "", label="(Optional) Name for Mlflow experiment run")
run_name = (
    None if dbutils.widgets.get("run_name") == "" else dbutils.widgets.get("run_name")
)

# COMMAND ----------
# DBTITLE 1,Define base and tuning parameters

# define prompts for tuning
system_prompt_1 = "You are a trustful assistant to people working in stores. You are answering questions based on the documents provided. If the provided documents do not contain relevant information, you truthfully say you do not know. Answer straight and do not introduce yourself.\n Here are some documents which might help you answer: {documents}"

# we use the default parameters for everything not specified as part of base_parameters or tuning-parameters
base_parameters = {
    "env": "dev",
    "experiment_name": "/Users/david.tempelmann@databricks.com/tuning_chatbot",
    "mlflow_run_name": "optimization_chunk_llm_7_vs_13",
}

# test 3 different chunk sizes and for each of the chunksizes two
# different overlaps (10% and 20% of chunk-size). In addition, we
# test two different LLMs
tuning_parameters = [
    {
        "chunk_overlap": "48",
        "chunk_size": "475",
        "top_k": "4",
        "llm_endpoint_name": "llama2_7b_chat_v3",
        "system_prompt": system_prompt_1,
    },
    {
        "chunk_overlap": "40",
        "chunk_size": "400",
        "top_k": "4",
        "llm_endpoint_name": "llama2_7b_chat_v3",
        "system_prompt": system_prompt_1,
    },
]


# COMMAND ----------
# DBTITLE 1,Run sequential grid-search for all defined tuning parameters


def notebook_path(relative_path: str):
    return str(Path(get_notebook_path()).parent.joinpath(relative_path).resolve())


client = mlflow.MlflowClient()
exp_id = client.get_experiment_by_name(experiment_name).experiment_id
parent_run = client.create_run(
    experiment_id=exp_id,
    run_name=run_name,
)
parent_run_id = {"mlflow_parent_run": parent_run.info.run_id}

job_ids = []
mlflow_run_ids = []
for params in tuning_parameters:
    parameters = base_parameters | params | parent_run_id

    silver_to_gold = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path("../../data/notebooks/03_silver_to_gold"),
            base_parameters=parameters,
        ),
        task_key="silver_to_gold",
    )

    gold_to_vs = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path("../../data/notebooks/04_gold_to_vs"),
            base_parameters=parameters,
        ),
        task_key="gold_to_vs",
        depends_on=[jobs.TaskDependency("silver_to_gold")],
    )

    create_and_eval_chatbot = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path("../../chatbot/notebooks/01_chatbot"),
            base_parameters=parameters,
        ),
        task_key="create_and_eval_chatbot",
        depends_on=[jobs.TaskDependency("gold_to_vs")],
    )

    job_run = w.jobs.submit(
        run_name="test_tuning",
        tasks=[silver_to_gold, gold_to_vs, create_and_eval_chatbot],
    ).result()

    # collect all performed job runs in a list
    job_ids.append(job_run.job_id)
    # collect mlflow run ids for later retrieval of experiment results
    child_run = client.search_runs(
        experiment_ids=exp_id,
        filter_string=f"tags.`mlflow.databricks.jobID` = '{job_run.job_id}'",
    )
    mlflow_run_ids.append(child_run[0].info.run_id)

# save some metadata to parent run
client.log_param(parent_run.info.run_id, "child_job_ids", json.dumps(job_ids))
client.log_param(
    parent_run.info.run_id, "child_mlflow_run_ids", json.dumps(mlflow_run_ids)
)
# finish the parent run
client.update_run(run_id=parent_run.info.run_id, status="FINISHED")

# COMMAND ----------

# DBTITLE 1,Load Results tables (artifacts from mlflow.evaluate) for all above experiment runs
experiment_results_pdf = client.load_table(
    exp_id,
    artifact_file="eval_results_table.json",
    run_ids=mlflow_run_ids,
    extra_columns=["run_id"],
)
experiment_results_pdf.display()
