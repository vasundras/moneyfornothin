import streamlit as st
from snowflake.snowpark import Session
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.custom import instrument, TruCustomApp
from trulens.providers.cortex.provider import Cortex
from trulens.core import Feedback, Select
from trulens.core import Tru
import numpy as np

# -------------------------
# Initialize Session State Defaults
# -------------------------
st.session_state.setdefault('model_name', 'mistral-large2')
st.session_state.setdefault('category_value', 'ALL')
st.session_state.setdefault('use_context', False)

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
# TruLens Integration 
# -------------------------
tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)
tru_session = TruSession(connector=tru_snowflake_connector)
tru_session.migrate_database()

# -------------------------
# Sidebar Configuration
# -------------------------
def config_options():
    st.sidebar.title("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    if "model_name" not in st.session_state:
        st.session_state.model_name = 'mistral-large2'
    selected_model = st.sidebar.selectbox(
        'Choose your model:',
        ['mistral-large2', 'mistral-large', 'mistral-7b'],
        index=['mistral-large2', 'mistral-large', 'mistral-7b'].index(st.session_state.model_name)
    )
    st.session_state.model_name = selected_model
    
    # Category Filter
    st.sidebar.subheader("Category Filter")
    if "category_value" not in st.session_state:
        st.session_state.category_value = 'ALL'
    try:
        categories = session.sql(
            "SELECT DISTINCT category FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE"
        ).collect()
        cat_list = ['ALL'] + [cat.CATEGORY for cat in categories]
    except Exception as e:
        st.sidebar.error(f"Failed to load categories: {e}")
        cat_list = ['ALL']
    selected_category = st.sidebar.selectbox(
        'Select a category:',
        cat_list,
        index=cat_list.index(st.session_state.category_value) if st.session_state.category_value in cat_list else 0
    )
    st.session_state.category_value = selected_category
    
    # Context Toggle
    st.sidebar.subheader("Context Toggle")
    if "use_context" not in st.session_state:
        st.session_state.use_context = False
    use_context = st.sidebar.checkbox(
        'Use document context?', 
        value=st.session_state.use_context
    )
    st.session_state.use_context = use_context
    
    # Data Source Toggle
    st.sidebar.subheader("Data Source")
    if "data_source" not in st.session_state:
        st.session_state.data_source = "IRS Data Only"
    selected_data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["IRS Data Only", "W-2 Data Only", "Both"],
        index=["IRS Data Only", "W-2 Data Only", "Both"].index(st.session_state.data_source)
    )
    st.session_state.data_source = selected_data_source

    # Testing Toggle
    st.sidebar.subheader("Testing")
    st.session_state.run_tests = st.sidebar.checkbox("Run Automated Tests")



# -------------------------
# Cortex Search Retriever Class
# -------------------------
class CortexSearchRetriever:
    def __init__(self, session, limit_to_retrieve=3):
        self.session = session
        self.limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query, category_value):
        if category_value == "ALL":
            response = self.session.sql(f"""
                SELECT * 
                FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
                LIMIT {self.limit_to_retrieve}
            """).collect()
        else:
            response = self.session.sql(f"""
                SELECT * 
                FROM IRS_PUBS_CORTEX_SEARCH_DOCS.DATA.DOCS_CHUNKS_TABLE 
                WHERE category = '{category_value}' 
                LIMIT {self.limit_to_retrieve}
            """).collect()
        return response

# -------------------------
# IRS RAG Class
# -------------------------
class IRS_RAG:
    def __init__(self, retriever, session):
        self.retriever = retriever
        self.session = session

    @instrument
    def retrieve_context(self, query: str):
        category_value = st.session_state.get("category_value", "ALL")
        return self.retriever.retrieve(query, category_value)

    @instrument
    def generate_completion(self, query: str, context_response: list) -> str:
        """
        Generate a response using Cortex Complete.

        Args:
            query (str): The user's question/query.
            context_response (list): Retrieved context chunks.

        Returns:
            str: The LLM-generated response.
        """
        if context_response:
            prompt_context = "\n\n".join([row.CHUNK for row in context_response])
            prompt = f"""
            You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.
            
            Below is CONTEXT extracted from IRS documents. Use it to assist in answering the QUESTION. 
            If the context does not fully answer the question, rely on your general knowledge of U.S. tax regulations.

            CONTEXT:
            {prompt_context}

            QUESTION:
            {query}

            INSTRUCTIONS:
            - Provide clear, concise, and authoritative answers.
            - Do not invent information.
            - If relevant, recommend IRS forms or publications.
            - If context is insufficient, fall back on your expert knowledge of the IRS taxation system.
            - If the nationality is Canadian, research the CRA website and the IRS website to provide a response.

            Answer:
            """
        else:
            prompt = f"""
            You are an expert IRS tax assistant with a deep understanding of IRS guidelines and general U.S. tax laws.

            QUESTION:
            {query}

            INSTRUCTIONS:
            - Provide clear, concise, and authoritative answers.
            - If relevant, recommend IRS forms or publications.

            Answer:
            """
        try:
            cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
            df_response = self.session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
            return df_response[0]['RESPONSE']
        except Exception as e:
            st.error(f"Failed to generate a response: {e}")
            return "Sorry, there was an error processing your question."

    @instrument
    def query(self, query: str) -> str:
        """
        Execute the full RAG pipeline: Retrieve context, create a prompt, and generate a response.

        Args:
            query (str): The user's question/query.

        Returns:
            str: The final response generated by the LLM.
        """
        context_response = self.retrieve_context(query)
        return self.generate_completion(query, context_response)

# -------------------------
# Initialize RAG
# -------------------------
retriever = CortexSearchRetriever(session)
rag = IRS_RAG(retriever, session)

# -------------------------
# Feedback Functions
# -------------------------
provider = Cortex(session, "mistral-large")

f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness").on_output()
f_context_relevance = Feedback(provider.context_relevance, name="Context Relevance").on_input().on_output().aggregate(np.mean)
f_answer_relevance = Feedback(provider.relevance, name="Answer Relevance").on_input().on_output().aggregate(np.mean)

tru_rag = TruCustomApp(
    app=rag,
    app_name="IRS_RAG_App",
    app_version="v1.0",
    feedbacks=[f_groundedness, f_context_relevance, f_answer_relevance]
)

# -------------------------
# TruLens Testing Prompts
# -------------------------
TEST_PROMPTS = [
    "What are the tax obligations for a Canadian living in the US?",
    "How do I claim the Foreign Earned Income Exclusion?",
    "What IRS forms do I need to file as a freelancer?",
    "How can I check my IRS refund status?",
    "What deductions can I claim as a freelancer?",
    "How do I report foreign bank accounts to the IRS?",
    "What is the difference between a 1040 and a 1040-NR form?",
    "How do I file taxes if I am on a visa in the US?",
    "What are the penalties for late IRS filings?",
    "How do I apply for an IRS payment plan?",
    "Which IRS form should I use for capital gains taxes?",
    "How do I determine my IRS filing status?",
    "Can I deduct home office expenses on my IRS tax return?",
    "What tax credits are available for higher education expenses?",
    "How do I handle IRS audits?",
    "What is the IRS standard deduction for 2024?",
    "Are medical expenses tax-deductible in the US?",
]

# -------------------------
# TruLens Automated Testing
# -------------------------
def run_tests():
    st.write("### Running Automated Tests with TruLens")
    with tru_rag as recording:
        for prompt in TEST_PROMPTS:
            try:
                response = rag.query(prompt)
                st.write(f"**Question:** {prompt}")
                st.write(f"**Response:** {response}")
            except Exception as e:
                st.error(f"Failed to process prompt: {prompt}. Error: {e}")
    
    # Display TruLens Leaderboard
    st.write("### TruLens Leaderboard")
    leaderboard = tru_session.get_leaderboard()
    st.dataframe(leaderboard)

# -------------------------
# Main App
# -------------------------
def main():
    st.title("Money for Nothin")
    st.markdown("*And Your Tax Advice For Free*")
    st.write("Ask tax-related questions and receive accurate answers sourced directly from IRS documents.")

    
    config_options()

    # Check if user enabled Automated Testing
    if st.session_state.get("run_tests", False):
        run_tests()
        return

    
    question = st.text_input("Ask a tax-related question:", placeholder="Enter your question here")
    if question:
        with st.spinner("Processing your question..."):
            response = rag.query(question)
        
        st.subheader("Response")
        st.write(response)

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    main()
