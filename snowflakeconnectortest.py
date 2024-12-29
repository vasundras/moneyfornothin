from snowflake.snowpark import Session
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession

# Connection Parameters
connection_params = {
    "account": "fwvflqp-xkb98144",
    "user": "VASUNDRA2308",
    "password": "Ramona123!",
    "warehouse": "LLMOPS_WH_M",
    "database": "IRS_PUBS_CORTEX_SEARCH_DOCS",
    "schema": "DATA",
    "role": "ACCOUNTADMIN"
}

# Create session
session = Session.builder.configs(connection_params).create()
print("✅ Snowflake Session Created Successfully")

# Test TruLens Connector
tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
tru_session = TruSession(connector=tru_snowflake_connector)
tru_session.migrate_database()
print("✅ TruLens Snowflake Connector Established Successfully")
