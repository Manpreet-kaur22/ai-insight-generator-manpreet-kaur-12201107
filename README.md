## Project Overview
The Conversational Insights Analyzer is an advanced AI-driven platform built to perform deep analysis of customer feedback, specifically focusing on large datasets like airline reviews (tweets). It moves beyond standard positive/negative labeling to provide Actionable Insights, Priority Action Plans, and Sentence-level Sentiment Flow visualization, enabling managers to make immediate, informed decisions.

## Problem Statement
Traditional sentiment analysis tools only offer shallow labels (positive/negative/neutral). Managers struggle to translate these scores into clear actionable next steps and often lack the ability to quickly drill down to the root causes of systemic issues.
Our Solution (Clone & Elevate):
We developed an AI-driven platform that elevates insight generation. It identifies high-priority issues across the dataset and provides clear, prioritized Action Plans and Root Cause Analysis (RCA) reports. The application features a comprehensive Dashboard with measurable KPIs and a Single Text Analyzer that dictates the immediate action required for every conversation.

## Solution Summary
The application is built using Streamlit for rapid prototyping and a modern UI, powered by Google's Gemini 2.5 Flash for deep natural language understanding and structured data generation.
The system operates in two core modes:
Dataset Dashboard (1_Dataset_Dashboard.py): Provides a high-level overview, including KPIs, Airline-wise Comparison, Prioritized Action Plans, and Root Cause Analysis (RCA) for system-wide issues identified from the negative feedback sample.
Single Text Analyzer (2_Single_Text_Analyzer.py): Offers real-time analysis of a pasted text, providing a Sentiment Score, Emotional Tone, Main Themes, Action Type, and a crucial Sentiment Flow line graph (visualization for conversational dynamics).
The use of Gemini's structured JSON output capabilities ensures the reliability and consistency of the analysis results displayed across the app.

## Tech Stack

1.The following technologies, frameworks, APIs, and tools were used in this project:

2.Backend/Frontend: Streamlit (Python framework for UI)

3.AI Models: Google Gemini 2.5 Flash (gemini-2.5-flash-preview-09-2025)

4.Data Processing: Pandas (Data manipulation)

5.Visualization: Plotly Express (Interactive charts for sentiment flow), WordCloud (Theme visualization)

6.Configuration: streamlit/secrets.toml (Secure API key storage)

7.Version Control: Git + GitHub

## Project Structure

Conversational_Insights_Analyzer/
├── pages/
│   ├── 1_Dataset_Dashboard.py    # Comprehensive dashboard with KPIs, RCA, Action Plan
│   └── 2_Single_Text_Analyzer.py # Real-time analysis of single conversation
├── app.py                        # Landing page for initialization and data loading
├── llm_service.py                # Core: LLMService class with all Gemini API logic
├── utils.py                      # Data loading utilities
├── requirements.txt              # All Python dependencies
└── README.md                     # This file

## Setup Instructions (with Conda)
Follow these steps precisely to run the project locally.
### 1. Clone the repository
    git clone https://github.com/<[your-repo-link](https://github.com/Manpreet-kaur22/ai-insight-generator-manpreet-kaur-12201107)>.git
    cd <repo-folder>

### 2. Install Dependencies

  Install all necessary Python packages listed in requirements.txt:
    pip install -r requirements.txt

### 3. Set Gemini API Key 
  The application securely loads the API key from Streamlit's secrets management.
  Create folder: If it doesn't exist, create a folder named .streamlit in the root directory.
  Create file: Inside .streamlit, create a file named secrets.toml.
  Add the key: Paste your Gemini API key into this file using the following format:
     GEMINI_API_KEY="YOUR_API_KEY_HERE"                                   Set Gemini API Key 

### 4. Run the Application
   Execute the main application file using Streamlit:
      streamlit run app.py
   The app will open automatically in your browser (typically at http://localhost:8501).


## Demo Video (Mandatory)
https://youtu.be/3qdGCQ9MBWo?si=aNr0aMMaOlVMRcQ-

## Features
Core LLM-Powered Highlights:.
- Actionable Insights: Generates a required Action Type (Escalate, Product Improvement, etc.) and a Justification for every conversation, directly assisting frontline staff.
- Root Cause Analysis (RCA): Provides a structured report identifying the top 3 underlying causes for poor overall performance (leveraging negative feedback samples).
- Sentiment Flow Visualization: Uses structured JSON output from Gemini to map the sentiment score of every sentence in a conversation, revealing how emotional tone changes over time.
- Prioritized Action Plan: Generates a markdown report with a clear plan focused on the worst-performing segments in the dataset.
- Conversational Data Insight: Allows users to ask natural language questions about the loaded dataset, receiving concise, data-backed answers.

## 🧩 Technical Architecture
The architecture relies on a modular Python structure with Streamlit serving as the full-stack UI.
> - Frontend sends query to the Streamlit Server (app.py / pages/)
> - Streamlit passes the request to the LLMService (in llm_service.py)
> - LLMService constructs a payload (including the user query, a strict System Instruction, and a JSON Schema for structured output)
> - LLMService calls the Gemini 2.5 Flash API using exponential backoff for reliability.
> - Gemini returns a structured JSON object or a Markdown report.
> - LLMService parses the output and returns the final data structure to the Streamlit UI.
> - Data is visualized (Plotly/WordCloud) or displayed (KPIs/Reports) in the web interface.

## 🧾 References

List any open-source resources, datasets, or models you used.
- [LLM API: Google Gemini API (gemini-2.5-flash-preview-09-2025))
- [Documentation: Streamlit Documentation)
- [Libraries: Pandas, Plotly Express)
- [Dataset Source: [[Insert Source of Airline Feedback Dataset here](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)])
