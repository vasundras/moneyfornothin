import streamlit as st  # UI Framework
from snowflake.snowpark.context import get_active_session  # Snowpark Session Management
from snowflake.cortex import Complete  # Direct Cortex LLM API
import pandas as pd  # Data Manipulation
import json  # JSON Parsing

# -------------------------
# Initialize Session State Defaults
# -------------------------
st.session_state.setdefault('model_name', 'mistral-large2')
st.session_state.setdefault('category_value', 'ALL')
st.session_state.setdefault('use_context', False)
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('slide_window', 5)  # Number of messages to keep in the window

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
# Establish Snowflake session
session = get_active_session()

# -------------------------
# Sidebar Configuration
# -------------------------
def config_options():
    st.sidebar.title("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    selected_model = st.sidebar.selectbox(
        'Choose your model:',
        ['mistral-large2', 'mistral-large', 'mistral-7b'],
        index=['mistral-large2', 'mistral-large', 'mistral-7b'].index(st.session_state.model_name)
    )
    st.session_state.model_name = selected_model
    
    # Category Filter
    st.sidebar.subheader("Category Filter")
    try:
        categories = session.sql(
            "SELECT DISTINCT category FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE"
        ).collect()
        cat_list = ['ALL'] + [cat['CATEGORY'] for cat in categories]
    except Exception as e:
        st.sidebar.error(f"Failed to load categories: {e}")
        cat_list = ['ALL']
    
    selected_category = st.sidebar.selectbox(
        'Select a category:',
        cat_list,
        index=0
    )
    st.session_state.category_value = selected_category
    
    # Context Toggle
    st.sidebar.subheader("Context Toggle")
    st.session_state.use_context = st.sidebar.checkbox('Use document context?', value=st.session_state.use_context)
    
    # Debug Toggle
    st.sidebar.subheader("Debug Toggle")
    st.session_state.debug = st.sidebar.checkbox('Enable Debug Mode', value=False)

# -------------------------
# Chat History Functions
# -------------------------
def get_chat_history():
    """
    Returns the last N messages from the chat history.
    """
    start_index = max(0, len(st.session_state.chat_history) - st.session_state.slide_window)
    return st.session_state.chat_history[start_index:]

def summarize_question_with_history(chat_history, question):
    """
    Summarizes the previous chat history and the current question.
    """
    prompt = f"""
    Summarize the following chat history and question to form a concise query:
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    """
    try:
        summary = Complete(model=st.session_state.model_name, prompt=prompt)
        return summary
    except Exception as e:
        st.error(f"Failed to summarize question with history: {e}")
        return question

# -------------------------
# Context Retrieval
# -------------------------
def get_similar_chunks_search_service(query):
    """
    Retrieve similar chunks from IRS_PUBS_CORTEX_SEARCH_DOCS.
    """
    try:
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
    except Exception as e:
        st.error(f"Failed to retrieve similar chunks: {e}")
        return []

# -------------------------
# Prompt Creation
# -------------------------
def create_prompt(question):
    """
    Creates the final LLM prompt using history, context, and the question.
    """
    chat_history = get_chat_history()
    
    if chat_history:
        question_summary = summarize_question_with_history(chat_history, question)
        context_chunks = get_similar_chunks_search_service(question_summary)
    else:
        context_chunks = get_similar_chunks_search_service(question)
    
    prompt_context = "\n\n".join([row.CHUNK for row in context_chunks])
    relative_paths = set(row.RELATIVE_PATH for row in context_chunks)
    
    prompt = f"""
    You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.
    
    CONTEXT:
    {prompt_context}

    QUESTION:
    {question}

    INSTRUCTIONS:
    - Provide clear, concise, and authoritative answers.
    - Do not invent information.
    - If relevant, recommend IRS forms or publications.
    - If context is insufficient, fall back on your expert knowledge.
    """
    return prompt, relative_paths

# -------------------------
# Answer Generation
# -------------------------
def answer_question(question):
    """
    Generate a response from the LLM.
    """
    prompt, relative_paths = create_prompt(question)
    try:
        response = Complete(model=st.session_state.model_name, prompt=prompt)
        return response, relative_paths
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "Sorry, there was an error processing your question.", []

# -------------------------
# Main App
# -------------------------
def main():
    st.title(":speech_balloon: Chat Document Assistant with Snowflake Cortex")
    st.write("Ask questions and receive accurate IRS tax advice using Snowflake Cortex and document search.")
    
    config_options()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User Input
    if question := st.chat_input("What do you want to know about IRS guidelines?"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response, relative_paths = answer_question(question)
                message_placeholder.markdown(response)
                
                if relative_paths:
                    with st.sidebar.expander("Related Documents"):
                        for path in relative_paths:
                            st.sidebar.markdown(f"📄 [Document Link: {path}](#)")
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    main()
