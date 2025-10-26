import streamlit as st
import pandas as pd
import plotly.express as px
from llm_service import LLMService

# --- Page Setup ---
st.title("🗣️ Analyze a Single Conversation")
st.caption("Uses Gemini to provide instant Sentiment, Tone, Themes, and Actionable Insights.")

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

# --- Session State Check ---
if 'llm_service' not in st.session_state or st.session_state.llm_service is None:
    st.error("🚨 LLM Service not initialized. Please go to the 'app' home page.")
    st.stop()
llm_service: LLMService = st.session_state.llm_service

# --- Input Area ---
text_to_analyze = st.text_area(
    "Paste Conversation Text Here:",
    key="single_input_text",
    # example : "This product is very good. But the price is too much high. It's confusing whether to buy or not."
    height=150,
    placeholder="Example: The instructions were great, but I wasted almost an hour on the final step. That was incredibly frustrating."
)
if st.button("Analyze Conversation", type="primary"):

    if text_to_analyze.strip():
        # Clear all previous analysis results from session state to force regeneration
        st.session_state['single_text_analysis'] = None
        st.session_state['sentiment_flow'] = pd.DataFrame()
        with st.spinner("Analyzing conversation with Gemini..."):
            # 1. Main Analysis Result
            result = llm_service.analyze_single_text(text_to_analyze)
            st.session_state['single_text_analysis'] = result
            # 2. Flow Analysis Result (For Graph)
            if 'error' not in result:
                flow_data = llm_service.analyze_sentiment_flow(text_to_analyze.strip())
                if flow_data:
                    try:
                        flow_df = pd.DataFrame(flow_data)
                        # Ensure column names are correctly set for plotting
                        flow_df.rename(columns={'sentence_number': 'Sentence_Number', 
                                                'sentiment_score': 'Sentiment_Score'}, inplace=True)
                        # CRITICAL FIX FOR GRAPH X-AXIS: 
                        # Convert Sentence_Number to integer then string for clean discrete x-axis labels
                        flow_df['Sentence_Number'] = flow_df['Sentence_Number'].astype(int).astype(str)
                        st.session_state.sentiment_flow = flow_df
                    except Exception as e:
                        st.error(f"Error processing flow data into DataFrame: {e}. Check LLM output structure.")
                        st.session_state.sentiment_flow = pd.DataFrame()
    else:
        st.warning("Please enter some text to analyze.")
        st.session_state['single_text_analysis'] = None

st.markdown("---")

# --- Display Results ---
analysis_result = st.session_state.get('single_text_analysis')

if analysis_result:
    if 'error' in analysis_result:
        st.error(f"Analysis Failed: {analysis_result['error']}")
    else:
        # 1. Main Analysis Results
        st.header("💡 Main Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Sentiment (Overall Feeling)")
            st.markdown(f"**Overall Polarity:** <span style='color:green;'>{analysis_result['sentiment_polarity']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Score:** `{analysis_result['sentiment_score']:.4f}`")
        with col2:
            st.subheader("2. Tone (Emotional Style)")
            st.markdown(f"**Emotion:** {analysis_result['emotional_tone']}")
        with col3:
            st.subheader("3. Theme (Main Topic)")
            st.markdown("**Top Keywords/Phrases:**")
            for theme in analysis_result['main_themes']:
                st.markdown(f"* {theme}")

        st.markdown("---")

        # 2. Actionable Insights
        st.header("🔥 Actionable Insights (Decision Support)")
        action_type = analysis_result['action_type']
        if "No Immediate Action" in action_type:
            action_color = "green"
        elif "Escalate" in action_type or "Service Recovery" in action_type:
            action_color = "red"
        elif "Product Improvement" in action_type:
            action_color = "orange"
        else:
            action_color = "blue"

        st.markdown(f"<p style='padding: 10px; border-left: 5px solid {action_color}; background-color: #f0f2f6;'>Action Type : {action_type}</p>", unsafe_allow_html=True)

        st.markdown(f"**Justification:** {analysis_result['justification']}")

        st.markdown("---")

        # 3. Visualization & Trend Analysis
        st.header("📉 Visualization & Trend Analysis")
        st.caption("This part demonstrates sentence-level trend analysis using Gemini.")
        flow_df = st.session_state.get('sentiment_flow', pd.DataFrame())
        # Plotting the result
        if not flow_df.empty and 'Sentence_Number' in flow_df.columns and 'Sentiment_Score' in flow_df.columns:
            st.subheader("Conversation Sentiment Flow")
            # Plotly Express Chart
            fig = px.line(
                flow_df, 
                x='Sentence_Number', # Now treated as discrete/categorical
                y='Sentiment_Score', 
                title='Emotional Flow Throughout the Conversation',
                markers=True # markers points nu dikhange
            )
            # Ensure the X-axis shows only integer labels (Sentence 1, 2, 3...)
            fig.update_xaxes(
                tickmode='array',
                tickvals=flow_df['Sentence_Number'].tolist(),
                title_text="Sentence Number"
            )
            fig.update_layout(yaxis_title="Sentiment Score")
            fig.add_hline(y=0, line_dash="dash", line_color="red") # Neutral line
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### How to Understand This Graph:")
            st.markdown("* **Y-Axis (Sentiment Score):** The zero line (0) is **Neutral**. The score indicates the emotional state of the speaker in each sentence.")
            st.markdown("* **Above 0:** The person is expressing **Positive** feelings (Happy/Excited).")
            st.markdown("* **Below 0:** The person is expressing **Negative** feelings (Upset/Frustrated).")
            st.markdown("* **Near 0:** The person is **Neutral** or stating a fact.")
            st.markdown("* **X-Axis (Sentence Number):** This shows the flow of the conversation from **Start (left) to End (right)**.")
        else:
            st.info("Could not generate sentiment flow data for this text. (Hint: Try entering more than one sentence!)")