import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import json
from trulens.core import Tru
from trulens.feedback.llm_provider import LLMProvider
from trulens.feedback.groundtruth import GroundTruthAggregator
from trulens.providers.cortex import CortexProvider

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
provider = CortexProvider()

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
    if st.session_state.use_context:
        context_response = get_similar_chunks(question)
        prompt_context = json.dumps([row.CHUNK for row in context_response])
        prompt = f"""
        You are an expert IRS tax advisor assistant that extracts information from the CONTEXT provided
        between <context> and </context> tags.
        When answering the question contained between <question> and </question> tags,
        be concise and do not hallucinate.
        If you don't have the information, just say so.
        Only answer the question if you can extract it from the CONTEXT provided.

        Recommend relevant tax forms when applicable.

        <context>
        {prompt_context}
        </context>
        <question>
        {question}
        </question>
        Answer:
        """
    else:
        prompt = f"Question: {question}\nAnswer:"
    
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
# Main App
# -------------------------
def main():
    st.title("Money for Nothin")
    st.markdown("*And Tax Advice for Free*")
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
        
        # TruLens Logging with LLMProvider
        provider = LLMProvider()
        ground_truth = GroundTruthAggregator()

        tru.record(
            question=question,
            response=response,
            metadata={
                "model": st.session_state.model_name,
                "use_context": st.session_state.use_context,
                "category": st.session_state.category_value
            },
            feedback=provider.evaluate_response(response)
        )
        
        # Display Related Document Link
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
