import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from llm_service import LLMService 

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

# --- Key Initialization ---
# Get API Key from secrets.toml (or environment variable if running locally)
if 'GEMINI_API_KEY' not in st.secrets:
    st.error("🚨 Configuration Error: 'GEMINI_API_KEY' not found in Streamlit secrets. Please ensure your secrets.toml file contains this key.")
    st.stop()
    
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']

# --- Page Setup ---
st.title("Airline Feedback Analytics Dashboard")
st.caption("Analyzes the complete dataset to identify company-wide issues.")

# --- Session State Check & LLM Service Initialization ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.info("⚠️ Dataset not loaded. Please go to the home page to load data.")
    st.stop()
    
# Initialize LLMService with the API Key
if 'llm_service' not in st.session_state:
    try:
        # Pass the retrieved API key to the LLMService constructor
        st.session_state.llm_service = LLMService(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"🚨 LLM Service Initialization Failed: {e}")
        st.stop()
llm_service: LLMService = st.session_state.llm_service


df: pd.DataFrame = st.session_state.df


# --- 1. Calculate KPIs ---
total_conversations = len(df)
# Safe check for Mean calculation
net_sentiment_score = df['Average Score'].mean() if 'Average Score' in df.columns else 0.0
escalation_count = df['Escalated'].sum() if 'Escalated' in df.columns else 0
escalation_rate = (escalation_count / total_conversations) * 100 if total_conversations > 0 else 0.0

st.markdown("---")
st.header("🎯 Key Performance Indicators (KPIs)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Conversations Analyzed", value=f"{total_conversations:,}")

with col2:
    st.metric(label="Net Sentiment Score (Avg)", value=f"{net_sentiment_score:.3f}")

with col3:
    st.metric(label="Escalation Rate", value=f"{escalation_rate:.1f}%", delta=f"↑ {escalation_count:,} tweets")

st.markdown("---")

# --- 2. Airline-wise Comparison (Table & Bar Chart) ---
st.header("✈️ Airline-wise Comparison")

# Check if essential columns are present before plotting
if 'airline' in df.columns and 'Average Score' in df.columns:
    # Calculate metrics for each airline
    airline_metrics = df.groupby('airline').agg(
        Average_Score=('Average Score', 'mean'),
        Total_Feedback=('airline', 'count')
    ).reset_index().sort_values(by='Average_Score', ascending=False)
    
    col4, col5 = st.columns([1, 2])

    with col4:
        st.subheader("Sentiment Ranking")
        # Display the ranking table
        airline_metrics['Average_Score'] = airline_metrics['Average_Score'].round(4)
        st.dataframe(airline_metrics.rename(columns={'Average_Score': 'Average Score', 'Total_Feedback': 'Total Feedback'}), 
                     hide_index=True, use_container_width=True)

    with col5:
        st.subheader("Average Sentiment by Airline")
        # Bar Chart using Plotly Express
        fig = px.bar(airline_metrics, x='airline', y='Average_Score', color='airline', 
                     title="Which Airline is Winning the Sentiment War?")
        fig.update_layout(xaxis_title="Airline", yaxis_title="Average Score")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Cannot display Airline Comparison. Data is missing 'airline' or 'Average Score' column.")


st.markdown("---")


# --- 3. Overall Sentiment Distribution (Pie Chart) ---
st.header("📊 Overall Sentiment Distribution")
st.caption("Categorizing the overall dataset sentiment using the Average Score.")

if 'Average Score' in df.columns:
    # 1. Define boundaries for sentiment
    def classify_sentiment(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    # 3. Apply the classification to the dataframe
    temp_df = df.copy() 
    temp_df['Sentiment Category'] = temp_df['Average Score'].apply(classify_sentiment)

    # 4. Count the occurrences of each category
    sentiment_counts = temp_df['Sentiment Category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # 5. Create the Pie Chart using Plotly Express
    fig_pie = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Percentage Breakdown of Conversation Sentiment',
        color='Sentiment',
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'red',
            'Neutral': 'orange'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.warning("Cannot display Sentiment Distribution. Data is missing 'Average Score' column.")

st.markdown("---")


# --- 4. Word Cloud (Top Themes) ---
st.header("☁️ Top Themes Across All Conversations")
st.caption("Visually identifying the most frequently mentioned words across all feedback.")

if 'text' in df.columns:
    # Concatenate all text and generate WordCloud
    all_text = " ".join(review for review in df['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Cannot generate Word Cloud. Data is missing 'text' column.")

st.markdown("---")


# --- 5. Root Cause Analysis (RCA) ---
st.header("🔍 Root Cause Analysis (RCA)")
st.caption("Gemini identifies the primary underlying reasons for poor sentiment and high escalation.")

# Button to trigger RCA generation
if st.button("Generate RCA Report", key="rca_button", type="secondary"):
    if st.session_state.df is not None and isinstance(st.session_state.df, pd.DataFrame):
        with st.spinner("Gemini is analyzing negative conversations and finding root causes..."):
            # Call the new RCA function
            rca_markdown = llm_service.generate_root_cause_analysis(df)
            st.session_state.rca_report = rca_markdown
    else:
        st.error("🚨 Error: Data not loaded or is invalid.")
        st.session_state.rca_report = "Error: Data is not available for RCA."

# Display the RCA report if generated
if 'rca_report' in st.session_state and st.session_state.rca_report:
    st.markdown(st.session_state.rca_report)
    
st.markdown("---")


# --- 6. Prioritized Action Plan Section (LLM Feature) ---
st.header("🔥 Prioritized Action Plan (Decision Support)")
st.caption("Gemini analyzes the KPIs and rankings to generate a top-priority action list for the operations team.")

if st.button("Generate Action Plan", type="primary"):
    if st.session_state.df is not None and isinstance(st.session_state.df, pd.DataFrame):
        df = st.session_state.df
        with st.spinner("Gemini is analyzing the data and crafting the action plan..."):
            action_plan_markdown = llm_service.generate_action_plan(df)
            st.session_state.action_plan = action_plan_markdown
    else:
        st.error("🚨 Error: DataFrame not loaded or is invalid. Please load data on the home page.")
        st.session_state.action_plan = "Error: Data is not available for analysis."


# Display the action plan if generated
if 'action_plan' in st.session_state and st.session_state.action_plan:
    st.markdown(st.session_state.action_plan)

st.markdown("---")


# --- 7. Conversational Data Insight (LLM Feature) ---
st.header("💬 Conversational Data Insight")
st.caption("Ask Gemini any specific question about the data ")

query = st.text_input("Ask a data question...", key="data_insight_query")

if st.button("Ask Gemini"):
    if query.strip():
        if st.session_state.df is not None and isinstance(st.session_state.df, pd.DataFrame):
            with st.spinner("Gemini is searching the data for insights..."):
                insight = llm_service.get_data_insight(df, query)
                st.session_state.data_insight_result = insight
        else:
            st.error("🚨 Error: Data not loaded. Please load data on the home page.")
            st.session_state.data_insight_result = "Error: Data is not available for analysis."
    else:
        st.warning("Please enter a question.")

# Display the insight result
if 'data_insight_result' in st.session_state and st.session_state.data_insight_result:
    st.markdown(st.session_state.data_insight_result)
