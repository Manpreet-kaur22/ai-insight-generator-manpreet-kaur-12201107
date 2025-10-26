import streamlit as st
import pandas as pd
from llm_service import LLMService  # LLM logic file
from utils import load_data        # Data loading utility
import os

# --- Page Configuration ---

st.set_page_config(

    page_title="Conversational Insights Analyzer",

    layout="wide",

    initial_sidebar_state="expanded"

)

# --- Custom CSS for sidebar ---
st.markdown("""
    <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #1E1E2F;  /* dark purple */
        }

        /* Sidebar text */
        [data-testid="stSidebar"] * {
            color: white !important;
            font-size: 18px;
        }

        /* Hover effect on sidebar items */
        [data-testid="stSidebarNav"] a:hover {
            color: #FFD700 !important;  /* golden hover */
            font-weight: bold;
        }

        /* Active link styling */
        [data-testid="stSidebarNav"] a.active {
            background-color: #4B0082 !important; /* indigo background */
            border-radius: 10px;
            padding: 5px 10px;
        }

        /* Sidebar title */
        [data-testid="stSidebarNav"]::before {
            content: "📊 Navigation";
            color: white;
            font-size: 22px;
            font-weight: bold;
            margin-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---

if 'df' not in st.session_state:

    st.session_state['df'] = None

if 'llm_service' not in st.session_state:

    st.session_state['llm_service'] = None

st.title("🗣️ Conversational Insights Analyzer")

st.caption("Analyzing Sentiment, Tone, and Themes in Public Feedback.")

st.markdown("---")


## 1. Initialize LLM Service (Gemini API)


api_key = st.secrets.get('GEMINI_API_KEY')

if not api_key:

    # API key not found error

    st.error("🚨 GEMINI_API_KEY not found in .streamlit/secrets.toml. Please add your key to run the analysis.")

else:

    try:

        # initialize LLMService only one time 

        if st.session_state['llm_service'] is None:

            st.session_state['llm_service'] = LLMService(api_key=api_key)

        st.success("✅ Gemini LLM Service Initialized successfully!")

    except Exception as e:

        st.error(f"Failed to initialize Gemini Client: {e}")

        st.session_state['llm_service'] = None


st.markdown("---")

## 2. Load Dataset (CSV or PKL)

uploaded_file = st.file_uploader(

    "Upload a new dataset (CSV or PKL).",

    type=["csv", "pkl"]

)

local_file_path = 'airline_analysis_data.pkl'

data_status_placeholder = st.empty()


# Data Loading Logic

if uploaded_file is not None:

    try:

        #  data loading completed from updated file

        st.session_state.df, file_name = load_data(uploaded_file)

        data_status_placeholder.success(f"✅ User file '{file_name}' Loaded: {len(st.session_state.df):,} conversations analyzed and ready.")

    except Exception as e:

        # Error handling for uploaded file

        data_status_placeholder.error(f"❌ Error loading uploaded file: {e}")

        st.session_state.df = None

elif st.session_state.df is None and os.path.exists(local_file_path):

    # load Local .pkl file (agar upload nahi hoya te session empty hai)

    try:

        st.session_state.df, file_name = load_data(None, file_path=local_file_path)

        data_status_placeholder.success(f"✅ Local dataset '{file_name}' Loaded: {len(st.session_state.df):,} conversations analyzed and ready.")

    except Exception as e:

        # Error handling for local file

        data_status_placeholder.error(f"❌ Error loading local file: {e}")

        st.session_state.df = None

elif st.session_state.df is not None:

    # Agar data pehlan hi loaded hai (page switch karan te)

    data_status_placeholder.success(f"✅ Dataset Loaded: {len(st.session_state.df):,} conversations analyzed and ready.")

else:

    data_status_placeholder.warning("⚠️ No dataset loaded. Please upload a file or ensure 'file_name.pkl' is in the folder.")


st.markdown("---")

## 3. Project Objective

st.header("Project Objective & Solution (Problem Statement)")

st.markdown("""

**Problem Statement:** Current tools only label feedback as positive/negative/neutral. Managers need **Actionable Insights**.



**Our Solution (Clone and Elevate):**

* **AI-Driven Platform** identifies issues and generates clear, prioritized **Action Plans**.

* **Dashboard** provides measurable KPIs and **Conversational Q&A** over the data.

* **Analyzer** provides Sentiment, Tone, Themes, and an **Immediate Action** for every conversation.

""")

