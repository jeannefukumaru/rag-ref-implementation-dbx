# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
from jeanne_chatbot.tuning.utils.query import query_endpoint, display_chat

# initialize conversation
messages = []

# COMMAND ----------
# DBTITLE 1,Query endpoint
# change the query to test different responses
query = "What is Apache Spark?"
# the query_endpoint function will query the endpoint and display the response
# it will also add the query and response to the messages list to simulate a
# multi-turn conversation. To start a new conversation, re-initialize the messages list.
messages = query_endpoint(query, messages, endpoint_name="jeanne_chatbot")