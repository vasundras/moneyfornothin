import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import json
from trulens.core import Tru
from trulens.feedback.llm_provider import LLMProvider

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Money for Nothin",
    page_icon="",
    layout="wide"
)

# -------------------------
# Initialize TruLens
# -------------------------
tru = Tru()
provider = LLMProvider(model_engine="mistral-large2")  # Default model set to mistral-large2

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

NUM_CHUNKS = 5
COLUMNS = ["chunk", "relative_path", "category"]

# -------------------------
# Sidebar Configuration
# -------------------------
def config_options():
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("Model Selection")
    st.sidebar.selectbox(
        'Choose your model:',
        ['mistral-large2', 'mistral-7b', 'mistral-large'],
        index=0,  # Default to 'mistral-large2'
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
# Retrieval Logic
# -------------------------
def get_similar_chunks(query):
    if st.session_state.category_value == "ALL":
        response = session.sql(f"""
            SELECT chunk, relative_path, category 
            FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
            LIMIT {NUM_CHUNKS}
        """).collect()
    else:
        response = session.sql(f"""
            SELECT chunk, relative_path, category 
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
        You are an IRS tax assistant with expertise in IRS documentation and tax guidelines.
        Below is some CONTEXT from IRS documents. Use it to answer the QUESTION.
        
        CONTEXT:
        {prompt_context}
        
        QUESTION:
        {question}
        
        Please provide clear, concise, and accurate information based on the context provided.
        If relevant, suggest specific IRS forms or publications.
        """
    else:
        prompt = f"Question: {question}\nAnswer: I'm sorry, but no relevant information was found in the context."
    
    return prompt


# -------------------------
# Completion Logic
# -------------------------
def complete(question):
    prompt = create_prompt(question)
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    
    if not df_response:
        return "No valid response received from the model."
    
    return df_response[0]['RESPONSE']


# -------------------------
# TruLens Logging
# -------------------------
def log_to_trulens(question, response):
    try:
        tru.add_record(
            app_id="moneyfornothin_app",  # Unique identifier for your app
            inputs={"question": question},
            outputs={"response": response},
            metadata={
                "model": st.session_state.model_name,
                "use_context": st.session_state.use_context,
                "category": st.session_state.category_value
            },
            calls=[{"input": question, "output": response}]
        )
    except Exception as e:
        st.error(f"TruLens Logging Failed: {e}")


# -------------------------
# Main App
# -------------------------
def main():
    st.title("Money for Nothin")
    st.markdown("*_And Your Tax Advice for Free_*")
    st.write("Ask tax-related questions and receive accurate answers sourced directly from IRS documents.")
    
    # Sidebar Configurations
    config_options()
    
    # Chat Interface
    question = st.text_input("Ask a tax-related question:", placeholder="Enter your question here")
    
    if question:
        with st.spinner("Processing your question..."):
            response = complete(question)
        
        st.subheader("Response")
        st.write(response)
        
        # Log to TruLens
        log_to_trulens(question, response)
        
        # Show Related Document Links
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
