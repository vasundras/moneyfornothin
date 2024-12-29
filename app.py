import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import json
import os
import threading
from trulens.core import Tru

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Money for Nothin",
    layout="wide"
)

# -------------------------
# Initialize TruLens
# -------------------------
tru = Tru()

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
# Constants
# -------------------------
IRS_COLORS = {
    "primary": "#004b87",
    "secondary": "#ffffff",
    "accent": "#ffd700"
}

NUM_CHUNKS = 3
COLUMNS = ["chunk", "relative_path", "category"]

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
    
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["IRS Data Only", "W-2 Data Only", "Both"]
    )
    
    return data_source

# -------------------------
# Retrieval Logic
# -------------------------
def get_similar_chunks(query):
    if st.session_state.category_value == "ALL":
        response = session.sql(f"""
            SELECT * 
            FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
            LIMIT {NUM_CHUNKS}
        """).collect()
    else:
        response = session.sql(f"""
            SELECT * 
            FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
            WHERE category = '{st.session_state.category_value}' 
            LIMIT {NUM_CHUNKS}
        """).collect()
    return response

# -------------------------
# Prompt Creation
# -------------------------
def create_prompt(question):
    context_response = get_similar_chunks(question)
    
    if context_response:
        prompt_context = "\n\n".join([row.CHUNK for row in context_response])
        prompt = f"""
        You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.
        
        Below is CONTEXT extracted from IRS documents. Use it to assist in answering the QUESTION. 
        If the context does not fully answer the question, rely on your general knowledge of U.S. tax regulations.

        CONTEXT:
        {prompt_context}

        QUESTION:
        {question}

        INSTRUCTIONS:
        - Provide clear, concise, and authoritative answers.
        - Do not invent information.
        - If relevant, recommend IRS forms or publications.
        - If context is insufficient, fall back on your expert knowledge of the IRS taxation system.
        - If the nationality is Canadian research the CRA website and the IRS website to provide a response.

        Answer:
        """
    else:
        prompt = f"""
        You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.

        QUESTION:
        {question}

        INSTRUCTIONS:
        - Provide clear, concise, and authoritative answers.
        - If relevant, recommend IRS forms or publications.

        Answer:
        """
    
    return prompt

# -------------------------
# Completion Logic
# -------------------------
def complete(question):
    prompt = create_prompt(question)
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    return df_response[0]['RESPONSE']

# -------------------------
# TruLens Logging
# -------------------------
def log_to_trulens(question, response, data_source):
    try:
        tru.add_record(
            app_id="moneyfornothin_app",
            inputs={"question": question},
            outputs={"response": response},
            metadata={
                "model": st.session_state.model_name,
                "use_context": st.session_state.use_context,
                "category": st.session_state.category_value,
                "data_source": data_source
            },
            calls=[
                {
                    "input": question,
                    "output": response,
                    "stack": "streamlit_app",
                    "args": {},
                    "pid": os.getpid(),
                    "tid": threading.get_ident()
                }
            ]
        )
    except Exception as e:
        st.error(f"TruLens Logging Failed: {e}")

# -------------------------
# Main App
# -------------------------
def main():
    st.title("Money for Nothin")
    st.markdown("*And Your Tax Advice For Free*")
    st.write("Ask tax-related questions and receive accurate answers sourced directly from IRS documents.")
    
    # Sidebar Configurations
    data_source = config_options()
    
    # Chat Interface
    question = st.text_input("Ask a tax-related question:", placeholder="Enter your question here")
    
    if question:
        with st.spinner("Processing your question..."):
            response = complete(question)
        
        st.subheader("Response")
        st.write(response)
        
        log_to_trulens(question, response, data_source)
        
        if st.session_state.use_context:
            st.subheader("Relevant IRS Publication")
            similar_chunks = get_similar_chunks(question)
            for chunk in similar_chunks:
                cmd = f"SELECT GET_PRESIGNED_URL(@docs, '{chunk.relative_path}', 360) AS URL_LINK"
                df_url = session.sql(cmd).to_pandas()
                st.markdown(f"[Read the full IRS Publication here]({df_url.iloc[0]['URL_LINK']})")

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    main()