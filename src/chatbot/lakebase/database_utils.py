import logging
from databricks.sdk import WorkspaceClient
import uuid
import psycopg

logger = logging.getLogger(__name__)

def build_db_uri(username, instance_name, database="databricks_postgres", port=5432):
    """
    Initialize a PostgreSQL connection pool for Databricks database instance.
    
    Args:
        username (str): Database username (e.g., "username@databricks.com")
        instance_name (str): Databricks database instance name (e.g., "demoinstance")
        database (str): Database name (default: "databricks_postgres")
        port (int): Database port (default: 5432)
        min_size (int): Minimum pool size (default: 1)
        max_size (int): Maximum pool size (default: 10)
    
    Returns:
        ConnectionPool: Configured connection pool with custom authentication
    """
    w = WorkspaceClient()
    
    # Get database instance details
    instance = w.database.get_database_instance(name=instance_name)
    host = instance.read_write_dns
    pgpassword = w.database.generate_database_credential(
                request_id=str(uuid.uuid4()), 
                instance_names=[host]).token
    
    username = username.replace("@", "%40")
    
    # Connection parameters
    db_uri = f"postgresql://{username}:{pgpassword}@instance-6f5de594-c934-4a4c-bf4e-8eba419382d6.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require"
    return db_uri

# Usage example:
# if __name__ == "__main__":
#     conn = build_db_uri(
#         username="<user_email>",
#         instance_name="demoinstance"
#     )
    
#     # Use the pool for database operations
#     try:
#         # Execute query
#         with conn.cursor() as cur:
#             cur.execute("SELECT version()")
#             version = cur.fetchone()[0]
#             print(version)
#     finally:
#         conn.close()  # Always close when done

