# Databricks notebook source
# MAGIC %pip install psycopg[binary] databricks-sdk==0.57

# COMMAND ----------

import sys
import os
sys.path.append("/Workspace/Users/jeanne.choo@databricks.com/jeanne_chatbot")

# COMMAND ----------

from chatbot.lakebase.database_utils import build_db_uri

# COMMAND ----------

dburi = build_db_uri(
        username="<user_email>",
        instance_name="demoinstance"
    )

import psycopg

# Connect using a PostgreSQL URI
with psycopg.connect(db_uri) as conn:
    try:
      with conn.cursor() as cur:
             cur.execute("select * from databricks_postgres.public.sample_table")
             version = cur.fetchall()
             print(version)
    finally:
         conn.close()  # Always close when done

# COMMAND ----------

