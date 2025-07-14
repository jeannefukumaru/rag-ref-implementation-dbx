# Databricks notebook source
# MAGIC %md 
# MAGIC # Mosaic AI Agent Framework: Author and deploy a tool-calling LangGraph agent
# MAGIC
# MAGIC This notebook demonstrates how to author a LangGraph agent that's compatible with Mosaic AI Agent Framework features. In this notebook you learn to:
# MAGIC - Author a tool-calling LangGraph agent wrapped with `ChatAgent`
# MAGIC - Manually test the agent's output
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow>=3.1.1 databricks-langchain uv langgraph==0.3.4 langgraph-checkpoint-postgres databricks-sdk==0.57 psycopg pytest

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC #### Agent tools
# MAGIC This agent code adds the built-in Unity Catalog function `system.ai.python_exec` to the agent. The agent code also includes commented-out sample code for adding a vector search index to perform unstructured data retrieval.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))
# MAGIC
# MAGIC #### Wrap the LangGraph agent using the `ChatAgent` interface
# MAGIC
# MAGIC For compatibility with Databricks AI features, the `LangGraphChatAgent` class implements the `ChatAgent` interface to wrap the LangGraph agent. This example uses the provided convenience APIs [`ChatAgentState`](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html#mlflow.langchain.chat_agent_langgraph.ChatAgentState) and [`ChatAgentToolNode`](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html#mlflow.langchain.chat_agent_langgraph.ChatAgentToolNode) for ease of use.
# MAGIC
# MAGIC Databricks recommends using `ChatAgent` as it simplifies authoring multi-turn conversational agents using an open source standard. See MLflow's [ChatAgent documentation](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent).
# MAGIC

# COMMAND ----------

dbutils.widgets.text("catalog_name", "users",)
catalog_name = dbutils.widgets.get("catalog_name")

dbutils.widgets.text("schema_name", "jeanne_choo",)
schema_name = dbutils.widgets.get("schema_name")

dbutils.widgets.text("project_name", "jeannes_chatbot",)
project_name = dbutils.widgets.get("project_name")

dbutils.widgets.text("llm_endpoint","databricks-claude-3-7-sonnet",)
llm_endpoint_name = dbutils.widgets.get("llm_endpoint")

dbutils.widgets.text("lakebase_name", "jeanne_lb")
lb_instance = dbutils.widgets.get("lakebase_name")

dbutils.widgets.text(
    "vector_search_index_name",
    "jeannes-chatbot_vs_index_direct",
)
# vs_index_name = dbutils.widgets.get("vector_search_index_name")
# vs_index_name = f"{catalog_name}.{schema_name}.{vs_index_name}"
vs_index_name = "main.dbdemos_rag_chatbot.databricks_documentation_vs_index"

# COMMAND ----------

with open("config.py", "w") as file:
    file.write(f'llm_endpoint_name = "{llm_endpoint_name}"\n')
    file.write(f'vs_index_name = "{vs_index_name}"\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register prompt in prompt registry

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks

# Register a prompt template
prompt = mlflow.genai.register_prompt(
    name=f"{catalog_name}.{schema_name}.system_prompt",
    template="you are a helpful assistant that can answer questions and perform tasks using tools",
    commit_message="Initial system prompt"
)
print(f"Created version {prompt.version}")  # "Created version 1"

# Set a production alias
mlflow.genai.set_prompt_alias(
    name=f"{catalog_name}.{schema_name}.system_prompt",
    alias="production",
    version=1
)

# COMMAND ----------

# %%writefile agent.py
from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.messages import convert_to_openai_messages
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import uuid
from lakebase.database_utils import build_db_uri
############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = ""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
# Below, we add the `system.ai.python_exec` UDF, which provides
# a python code interpreter tool to our agent
# You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools

# TODO: Add additional tools
uc_tool_names = ["system.ai.python_exec"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# for details

# TODO: Add vector search indexes
vector_search_tools = [
        VectorSearchRetrieverTool(
        index_name=vs_index_name,
        # filters="..."
    )
]
tools.extend(vector_search_tools)

#####################
## Define agent logic
#####################

class LangGraphChatAgent(ChatAgent):
    """
    A chat agent that uses LangGraph and Databricks tools for conversational AI with memory and tool use.
    """

    def __init__(self, model: LanguageModelLike, tools: list[BaseTool], db_uri: str) -> None:
        """
        Initialize the LangGraphChatAgent.

        Args:
            model (LanguageModelLike): The language model to use for generating responses.
            tools (list[BaseTool]): List of tools available to the agent.
            db_uri (str): Database URI for memory and checkpointing.
        """
        self.model = model
        self.tools = tools
        self.agent = None
        self.db_uri = db_uri

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        Generate a response to a sequence of chat messages.

        Args:
            messages (list[ChatAgentMessage]): List of chat messages in the conversation.
            context (Optional[ChatContext]): Optional chat context.
            custom_inputs (Optional[dict[str, Any]]): Optional custom inputs.

        Returns:
            ChatAgentResponse: The agent's response containing a list of messages.
        """
        with (PostgresStore.from_conn_string(db_uri) as store,
              PostgresSaver.from_conn_string(db_uri) as checkpointer,):
        
            self.agent = create_react_agent(self.model, self.tools)

            request = {"messages": self._convert_messages_to_dict(messages)}

            messages = []
            for event in self.agent.stream(request, stream_mode="updates"):
                for node_data in event.values():
                    messages.extend(
                        msg for msg in node_data.get("messages", [])
                    )
        messages = [convert_to_openai_messages(m) for m in messages]
        messages = [ChatAgentMessage(**msg, id=str(uuid.uuid4())) for msg in messages]
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Stream responses to a sequence of chat messages.

        Args:
            messages (list[ChatAgentMessage]): List of chat messages in the conversation.
            context (Optional[ChatContext]): Optional chat context.
            custom_inputs (Optional[dict[str, Any]]): Optional custom inputs.

        Yields:
            Generator[ChatAgentChunk, None, None]: Stream of chat agent chunks.
        """
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": convert_to_openai_messages(msg)}, id=str(uuid.uuid4())) for msg in node_data["messages"]
                )
    def __init__(self, model, tools, db_uri):
        self.model = model
        self.tools = tools
        self.agent = None
        self.db_uri = db_uri

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        with (PostgresStore.from_conn_string(db_uri) as store,
              PostgresSaver.from_conn_string(db_uri) as checkpointer,):
        
            self.agent = create_react_agent(self.model, self.tools)

            request = {"messages": self._convert_messages_to_dict(messages)}

            messages = []
            for event in self.agent.stream(request, stream_mode="updates"):
                for node_data in event.values():
                    messages.extend(
                        msg for msg in node_data.get("messages", [])
                    )
        messages = [convert_to_openai_messages(m) for m in messages]
        messages = [ChatAgentMessage(**msg, id=str(uuid.uuid4())) for msg in messages]
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": convert_to_openai_messages(msg)}, id=str(uuid.uuid4())) for msg in node_data["messages"]
                )

mlflow.langchain.autolog()
# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
db_uri = build_db_uri(username="jeanne.choo@databricks.com", instance_name="jeannelb")
AGENT = LangGraphChatAgent(model, tools, db_uri)
mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output and tool-calling abilities. Since this notebook called `mlflow.langchain.autolog()`, you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

# from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "What is Databricks?"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 5+5 in python"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
from agent import tools
from config import llm_endpoint_name
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

resources = [DatabricksServingEndpoint(endpoint_name=llm_endpoint_name)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        code_paths=['./config.py'],
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-vectorsearch=={get_distribution('databricks-vectorsearch').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with Agent Evaluation
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics.
# MAGIC See Databricks documentation ([AWS]((https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/mlflow3/genai/eval-monitor/custom-scorers) | [Azure](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/custom-scorers)).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

eval_dataset = [
    {
        "inputs": {"messages": [{"role": "user", "content": "What is an LLM?"}]},
        "expected_response": None,
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda messages: AGENT.predict({"messages": messages}),
    scorers=[RelevanceToQuery(), Safety()],
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{project_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, endpoint_name="agents_users-jeanne_choo-jeanne_chatbot", tags = {"endpointSource": "docs"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)).