from snowflake.snowpark import Session
from snowflake.cortex import Complete
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Snowpark session
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
}

try:
    session = Session.builder.configs(connection_params).create()
    print("✅ Snowpark Session Created Successfully!")
except Exception as e:
    print(f"❌ Failed to create Snowpark session: {e}")
    exit(1)

# Test Cortex Complete
try:
    response = Complete("mistral-large", "How do snowflakes get their unique patterns?", session=session)
    print("✅ Cortex Response:", response)
except Exception as e:
    print(f"❌ Cortex Complete Failed: {e}")
