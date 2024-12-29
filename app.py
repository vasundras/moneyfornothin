import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import json
import os
import threading

# TruLens Imports
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.custom import instrument, TruCustomApp
from trulens.providers.cortex.provider import Cortex
from trulens.core import Feedback, Select
import numpy as np

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Money for Nothin",
    layout="wide"
)

# -------------------------
# Snowflake Connection
# -------------------------
@st.cache_resource
def get_session():
    return Session.builder.configs({
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"]
    }).create()

session = get_session()

# -------------------------
# TruLens Configuration
# -------------------------
tru_connector = SnowflakeConnector(snowpark_session=session)
tru_session = TruSession(connector=tru_connector)

# -------------------------
# RAG Implementation
# -------------------------
class TaxAdvisorRAG:
    def __init__(self):
        self.retriever = self.setup_retriever()
    
    def setup_retriever(self):
        return session.sql(f"""
            SELECT * 
            FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
            LIMIT 3
        """).collect()
    
    @instrument
    def retrieve_context(self, query: str) -> list:
        """
        Retrieve relevant text from the IRS documents.
        """
        if st.session_state.category_value == "ALL":
            response = session.sql(f"""
                SELECT * 
                FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
                LIMIT 3
            """).collect()
        else:
            response = session.sql(f"""
                SELECT * 
                FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
                WHERE category = '{st.session_state.category_value}' 
                LIMIT 3
            """).collect()
        return response

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate a response using the context retrieved from IRS documents.
        """
        prompt = f"""
        You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.

        CONTEXT:
        {context_str}

        QUESTION:
        {query}

        INSTRUCTIONS:
        - Provide clear, concise, and authoritative answers.
        - Recommend IRS forms or publications if applicable.
        - If the context doesn't have sufficient information, rely on general IRS knowledge.
        - If the nationality is Canadian, please cross reference CRA knowledge with IRS knowledge. 
        
        """
        cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
        df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
        return df_response[0]['RESPONSE']

    @instrument
    def query(self, query: str) -> str:
        """
        Complete the query using retrieved context.
        """
        context = self.retrieve_context(query)
        context_str = "\n\n".join([row.CHUNK for row in context])
        return self.generate_completion(query, context_str)


# Instantiate RAG
rag = TaxAdvisorRAG()

# -------------------------
# Feedback Functions
# -------------------------
provider = Cortex(session.connection, "mistral-large")

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve_context.rets[:].collect())
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve_context.rets[:])
    .aggregate(np.mean)
)

f_answer_relevance = (
    Feedback(provider.relevance, name="Answer Relevance")
    .on_input()
    .on_output()
    .aggregate(np.mean)
)

# Register Feedback with TruLens
tru_rag = TruCustomApp(
    rag,
    app_name="MoneyForNothin",
    app_version="v1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# -------------------------
# Sidebar Configuration
# -------------------------
def config_options():
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("Model Selection")
    st.sidebar.selectbox(
        'Choose your model:',
        ['mistral-7b', 'mistral-large', 'mistral-large2'],
        key="model_name"
    )
    
    st.sidebar.subheader("Category Filter")
    categories = session.sql("SELECT DISTINCT category FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE").collect()
    cat_list = ['ALL'] + [cat.CATEGORY for cat in categories]
    st.sidebar.selectbox(
        'Select a category:',
        cat_list,
        key="category_value"
    )
    
    st.sidebar.subheader("Context Toggle")
    st.sidebar.checkbox('Use document context?', key='use_context')

# -------------------------
# Main App
# -------------------------
def main():
    st.title("Money for Nothin")
    st.markdown("*And Your Tax Advice For Free*")
    st.write("Ask tax-related questions and receive accurate answers sourced directly from IRS documents.")
    
    config_options()
    
    question = st.text_input("Ask a tax-related question:", placeholder="Enter your question here")
    
    if question:
        with tru_rag as recording:
            with st.spinner("Processing your question..."):
                response = rag.query(question)
            
            st.subheader("Response")
            st.write(response)
            
            if st.session_state.use_context:
                st.subheader("Relevant IRS Publication")
                similar_chunks = rag.retrieve_context(question)
                for chunk in similar_chunks:
                    cmd = f"SELECT GET_PRESIGNED_URL(@docs, '{chunk.relative_path}', 360) AS URL_LINK"
                    df_url = session.sql(cmd).to_pandas()
                    st.markdown(f"[Read the full IRS Publication here]({df_url.iloc[0]['URL_LINK']})")

    # Show TruLens Leaderboard
    leaderboard = tru_session.get_leaderboard()
    st.subheader("TruLens Leaderboard")
    st.write(leaderboard)

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    main()
