import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import json

# -------------------------
# 🎯 Configuration
# -------------------------
IRS_COLORS = {
    "primary": "#004b87",
    "secondary": "#ffffff",
    "accent": "#ffd700"
}

# Constants
NUM_CHUNKS = 3
COLUMNS = ["chunk", "relative_path", "category"]

# Snowflake Connection
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
svc = session.sql("SELECT * FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE")

# -------------------------
# 🎯 Sidebar Configuration
# -------------------------
def config_options():
    st.sidebar.title("🔧 Configuration")
    st.sidebar.selectbox(
        'Select your model:',
        ['mistral-7b', 'mistral-large', 'mistral-large2'],
        key="model_name"
    )
    
    categories = session.sql("SELECT DISTINCT category FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE").collect()
    cat_list = ['ALL'] + [cat.CATEGORY for cat in categories]
    st.sidebar.selectbox(
        'Select a category:',
        cat_list,
        key="category_value"
    )
    
    st.sidebar.checkbox('Use document context?', key='use_context')

# -------------------------
# 🎯 Retrieval Logic
# -------------------------
def get_similar_chunks(query):
    if st.session_state.category_value == "ALL":
        response = session.sql(f"SELECT * FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE LIMIT {NUM_CHUNKS}").collect()
    else:
        response = session.sql(f"SELECT * FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE WHERE category='{st.session_state.category_value}' LIMIT {NUM_CHUNKS}").collect()
    return response

# -------------------------
# 🎯 Prompt Creation
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
# 🎯 Completion Logic
# -------------------------
def complete(question):
    prompt = create_prompt(question)
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    return df_response[0]['RESPONSE']

# -------------------------
# 🎯 Main App
# -------------------------
def main():
    st.set_page_config(
        page_title="IRS Tax Chat Assistant",
        page_icon="🧾",
        layout="wide"
    )
    
    st.title("🧾 IRS Tax Chat Assistant")
    st.write("Ask tax-related questions and receive accurate answers sourced directly from IRS documents.")
    
    config_options()
    
    st.write("### 📑 Available Documents")
    docs_available = session.sql("LS @docs").collect()
    st.write(pd.DataFrame([doc["name"] for doc in docs_available], columns=["Document Name"]))
    
    question = st.text_input("Ask a tax-related question:", placeholder="Enter your question here")
    
    if question:
        with st.spinner("Thinking..."):
            response = complete(question)
        
        st.markdown("### 🤖 **Answer:**")
        st.markdown(response)

# -------------------------
# 🎯 Run the App
# -------------------------
if __name__ == "__main__":
    main()
