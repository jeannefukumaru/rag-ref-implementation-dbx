# Databricks notebook source
# MAGIC %pip install psycopg[binary] databricks-sdk==0.57

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Create lakebase instance

# COMMAND ----------

import requests

# replace with your own secret scope
databricks_token = dbutils.secrets.get(scope="demo-scope", key="pat")
instance_name = "demoinstance"
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
sp = "eb61a0e5-195f-44d2-985b-e7507d433c9e"


payload = {
    "name": instance_name,
    "capacity": "CU_1",
    "retention_window_in_days": 14
}

headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{workspace_url}/api/2.0/database/instances",
    headers=headers,
    json=payload
)

display(response.json())

# COMMAND ----------

# MAGIC %md 
# MAGIC # Create UC Catalog inside lakebase instance

# COMMAND ----------

import os
import requests

catalog_name = "demo_lakebase"
pg_database_name = "databricks_postgres"
create_database_if_not_exists = False  
instance_name = "demoinstance"  
payload = {
    "name": catalog_name,
    "database_name": pg_database_name,
    "instance_name": instance_name,
    "create_database_if_not_exists": create_database_if_not_exists
}

headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{workspace_url}/api/2.0/database/catalogs",
    headers=headers,
    json=payload
)

display(response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC # Configure permissions

# COMMAND ----------

from sqlalchemy import create_engine, text

from databricks.sdk import WorkspaceClient
import uuid

w = WorkspaceClient()
sp = "eb61a0e5-195f-44d2-985b-e7507d433c9e"
# sp = "1ba2c472-d8c6-48c3-b286-81443b299e44"
instance_name = "demoinstance"

instance = w.database.get_database_instance(name=instance_name)
cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[instance_name])

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
host = instance.read_write_dns
port = 5432
database = "databricks_postgres"
password = cred.token

connection_pool = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require")

with connection_pool.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    for row in result:
        print(f"Connected to PostgreSQL database. Version: {row}")

# COMMAND ----------

# DBTITLE 1,Create extension to handle permissions
with connection_pool.connect() as conn:
    result = conn.execute(text("CREATE EXTENSION IF NOT EXISTS databricks_auth;"))
    for row in result:
      print(row)

# COMMAND ----------

sp = "eb61a0e5-195f-44d2-985b-e7507d433c9e"
sql_string = f"SELECT databricks_create_role('{sp}', 'SERVICE_PRINCIPAL');"
with connection_pool.connect() as conn:
    result = conn.execute(text(sql_string))
    for row in result:
      print(row)

# COMMAND ----------

sql_string = f'GRANT ALL PRIVILEGES ON SCHEMA public TO "{sp}";'
with connection_pool.connect() as conn:
    result = conn.execute(text(sql_string))
    print(result)

# COMMAND ----------

with connection_pool.connect() as conn:
    result = conn.execute(text("SELECT * from databricks_list_roles;"))
    for row in result:
        print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC # UC Catalog permissions setup on catalog tied to database

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT ALL PRIVILEGES ON CATALOG demo_lakebase TO `eb61a0e5-195f-44d2-985b-e7507d433c9e`;

# COMMAND ----------

