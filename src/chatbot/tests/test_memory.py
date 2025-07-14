# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow>=3.1.1 databricks-langchain uv langgraph==0.3.4 langgraph-checkpoint-postgres databricks-sdk==0.57 psycopg pytest

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Test long-term memory

# COMMAND ----------

from config import llm_endpoint_name, instance_name

# COMMAND ----------

from lakebase.database_utils import build_db_uri
from databricks_langchain import ChatDatabricks
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import uuid

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

db_uri = build_db_uri("jeanne.choo@databricks.com", "demoinstance")

with (PostgresSaver.from_conn_string(db_uri) as checkpointer,
      PostgresStore.from_conn_string(db_uri) as store):
  store.setup()
  checkpointer.setup()
  model = ChatDatabricks(endpoint=llm_endpoint_name)

  def call_model(
      state: MessagesState,
      config: RunnableConfig,
      *,
      store: BaseStore,
  ):
      user_id = config["configurable"]["user_id"]
      namespace = ("memories", user_id)
      memories = store.search(namespace, query=str(state["messages"][-1].content))
      info = "\n".join([d.value["data"] for d in memories])
      system_msg = f"You are a helpful assistant talking to the user. User info: {info}"
      # Store new memories if the user asks the model to remember
      last_message = state["messages"][-1]
      if "remember" in last_message.content.lower():
          memory = "User name is Bob"
          store.put(namespace, str(uuid.uuid4()), {"data": memory})
      response = model.invoke(
          [{"role": "system", "content": system_msg}] + state["messages"]
      )
      print(response)
      return {"messages": response.content}

  builder = StateGraph(MessagesState)
  builder.add_node(call_model)
  builder.add_edge(START, "call_model")
  graph = builder.compile(
      checkpointer=checkpointer,
      store=store,
  )
  config = {
      "configurable": {
          "thread_id": "1",
          "user_id": "1",
      }
  }
  for chunk in graph.stream(
      {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
      config,
      stream_mode="values",
  ):
      chunk["messages"][-1]
  config = {
      "configurable": {
          "thread_id": "2",
          "user_id": "1",
      }
  }
  for chunk in graph.stream(
      {"messages": [{"role": "user", "content": "what is my name?"}]},
      config,
      stream_mode="values",
  ):
      chunk["messages"][-1]

# COMMAND ----------

