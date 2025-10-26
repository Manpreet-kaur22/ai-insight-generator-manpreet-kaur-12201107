
import pandas as pd

from io import BytesIO

import pickle

from typing import Tuple, Optional

import numpy as np

import streamlit as st 

import os



def load_data(uploaded_file: Optional[BytesIO], file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:

    """Loads data from a Streamlit uploaded file or a local file path (for pre-loaded data)."""

    

    # 1. Source te File Name Decide Karo

    file_source = None

    if uploaded_file:

        file_name = uploaded_file.name

        file_source = BytesIO(uploaded_file.getvalue())

    elif file_path and os.path.exists(file_path):

        file_name = file_path.split('/')[-1]

        file_source = file_path

    else:

        raise ValueError("No valid file source provided.")



    df = pd.DataFrame()

    

    # 2. DataFrame Load Karo (CSV ya PKL)

    if file_name.endswith('.csv'):

        try:

            # CSV: try utf-8, then latin-1

            if isinstance(file_source, str):

                df = pd.read_csv(file_source, encoding='utf-8')

            else:

                df = pd.read_csv(file_source, encoding='utf-8')

        except UnicodeDecodeError:

            if isinstance(file_source, str):

                df = pd.read_csv(file_source, encoding='latin-1')

            else:

                file_source.seek(0) # Reset pointer

                df = pd.read_csv(file_source, encoding='latin-1')

            

    elif file_name.endswith('.pkl'):

        try:

            # PKL: try simple read_pickle, then standard pickle load

            # Error Fix: The memo value error suggests the PKL file might be created with a different Python/Pandas version.

            # pd.read_pickle is usually the most robust method.

            if isinstance(file_source, str):

                df = pd.read_pickle(file_source)

            else:

                df = pd.read_pickle(file_source)

        except Exception as e:

            # More informative error message for PKL issue

            raise ValueError(f"Could not read PKL file. Error: {e}. Please try converting the data to a fresh CSV/PKL.")

    else:

        raise ValueError("Unsupported file format. Please use a CSV or PKL file.")



    if len(df) == 0:

        raise ValueError("The loaded DataFrame is empty.")



    # 3. Zaroori Columns da Check & Dummy Values Assign Karo

    

    # 'airline' column (Zaroori for Dashboard)

    if 'airline' not in df.columns:

        df['airline'] = 'Unknown Airline' # Safe default value

        st.warning("Warning: 'airline' column not found. Used 'Unknown Airline'.", icon="⚠️")



    # A) 'Average Score' (KPI/Ranking)

    if 'Average Score' not in df.columns:

        st.warning("Warning: 'Average Score' not found. Creating a dummy 'Average Score' column.", icon="⚠️")

        if 'airline_sentiment' in df.columns:

            # Use sentiment column if available

            sentiment_map = {'positive': 0.5, 'neutral': 0.0, 'negative': -0.5}

            df['Average Score'] = df['airline_sentiment'].map(sentiment_map).fillna(np.random.uniform(-0.5, 0.5))

        else:

             df['Average Score'] = np.random.uniform(-0.5, 0.5, len(df))

             

    # B) 'Escalated' column (KPI)

    if 'Escalated' not in df.columns:

        st.warning("Warning: 'Escalated' not found. Creating dummy 'Escalated' column.", icon="⚠️")

        # Example logic: Negative score below -0.2 is an escalation

        df['Escalated'] = (df['Average Score'] < -0.2).astype(int) # Convert to int for sum

        

    # C) 'text' column (WordCloud/LLM)

    if 'text' not in df.columns:

        if 'cleaned_text' in df.columns: # Jiven ki Screenshot 2025-10-24 222837.png vich hai

            df['text'] = df['cleaned_text']

            st.warning("Warning: 'text' column not found. Using 'cleaned_text' instead.", icon="⚠️")

        else:

            df['text'] = "Sample text for analysis"

            st.warning("Warning: 'text' column not found. Using sample text.", icon="⚠️")

            

    # Ensure text column is string for processing

    df['text'] = df['text'].astype(str)



    return df, file_name



