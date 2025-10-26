import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
import requests
import os

# --- Constants ---
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# --- Utility Functions for API Call and Exponential Backoff ---

def _call_gemini_api(payload: Dict[str, Any], api_key: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """Makes a POST request to the Gemini API with exponential backoff."""
    # API Key is passed as an argument
    url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                # Handle Rate Limit (429) with exponential backoff
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            # Re-raise for other HTTP errors or if max retries reached
            raise e
        except Exception:
            if attempt < max_retries - 1:
                # Handle connection/network errors
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise

    return None

def _extract_text_from_response(response: Dict[str, Any]) -> str:
    """Extracts the text content from the Gemini API response."""
    try:
        return response['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        return "Error: Could not extract text from the model response."


# --- LLMService Class ---

class LLMService:
    def __init__(self, api_key: str):
        # Store the API key in the instance
        self.api_key = api_key
        pass

    def _execute_api_call(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Internal helper to call the Gemini API using the stored key."""
        return _call_gemini_api(payload, self.api_key)


    def generate_action_plan(self, df: pd.DataFrame) -> str:
        """Generates a prioritized action plan based on airline metrics."""
        
        if 'airline' not in df.columns or 'Average Score' not in df.columns:
            return "Error: DataFrame is missing required 'airline' or 'Average Score' columns for Action Plan generation."

        airline_metrics = df.groupby('airline').agg(
            Average_Score=('Average Score', 'mean'),
            Total_Feedback=('airline', 'count')
        ).reset_index().sort_values(by='Average_Score', ascending=True) 

        worst_performers = airline_metrics.head(3).to_markdown(index=False)
        overall_stats = df['Average Score'].describe().to_markdown()

        system_prompt = (
            "You are a Senior Airline Operations Consultant. Your task is to review the provided sentiment data "
            "and generate a clear, highly prioritized Action Plan in markdown format. "
            "The plan must focus on the airlines with the WORST Average Score (lowest performers). "
            "Use clear headings, bullet points, and an executive summary. "
            "Structure the response with sections: 'Executive Summary', 'Priority Actions (Top 3 Airlines)', 'Key Recommendation'."
        )

        user_query = (
            f"Here are the worst-performing airlines by Average Sentiment Score:\n\n"
            f"{worst_performers}\n\n"
            f"Overall Sentiment Statistics:\n{overall_stats}\n\n"
            "Generate the prioritized action plan based on these metrics."
        )

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }

        response = self._execute_api_call(payload)
        return _extract_text_from_response(response) if response else "Error in Action Plan generation."


    def get_data_insight(self, df: pd.DataFrame, query: str) -> str:
        """Answers a user's conversational question about the data."""
        
        relevant_cols = [col for col in ['text', 'airline', 'Average Score', 'sentiment', 'Escalated'] if col in df.columns]
        
        if not relevant_cols:
             return "Error: DataFrame lacks sufficient columns for conversational analysis."

        sample_size = min(100, len(df)) 
        data_sample = df[relevant_cols].sample(n=sample_size, random_state=42).to_markdown(index=False)

        system_prompt = (
            "You are an expert Data Analyst specializing in airline feedback. "
            "Analyze the provided sample data and context to answer the user's question. "
            "Provide a concise, insightful explanation in a friendly and professional tone. "
            "If the answer requires a specific data point, state the observation based on the provided data sample. "
            "Start with 'Insight from Data:'."
        )

        user_query = (
            f"Analyze the data below and answer the following question:\n\n"
            f"User Question: '{query}'\n\n"
            f"Data Sample (Snapshot of relevant tweets):\n{data_sample}\n\n"
            "Provide the insight."
        )

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "tools": [{"google_search": {}}], 
        }

        response = self._execute_api_call(payload)
        return _extract_text_from_response(response) if response else "Error in Data Insight generation."
    
    
    def generate_root_cause_analysis(self, df: pd.DataFrame) -> str:
        """
        Generates a Root Cause Analysis (RCA) based on the most negative feedback 
        and high-level metrics.
        """
        
        # 1. High-level Metrics (KPIs)
        total_conversations = len(df)
        net_sentiment_score = df['Average Score'].mean() if 'Average Score' in df.columns else 0.0
        
        # 2. Worst Performing Airline for RCA focus
        if 'airline' in df.columns and 'Average Score' in df.columns:
            worst_airline = df.groupby('airline')['Average Score'].mean().idxmin()
        else:
            worst_airline = "N/A (Missing data)"


        # 3. Sample of most negative tweets (The "problem" evidence)
        if 'Average Score' in df.columns and 'text' in df.columns:
            # Filter for the most negative 
            negative_df = df[df['Average Score'] < -0.5]
            
            # Take a random sample of the most negative tweets for analysis context
            sample_size = min(50, len(negative_df))
            negative_sample = negative_df[['airline', 'text', 'Average Score']].sample(n=sample_size, random_state=42).to_markdown(index=False)
        else:
            negative_sample = "Negative sentiment text sample is unavailable."


        system_prompt = (
            "You are a highly experienced Root Cause Analyst for the aviation industry. "
            "Analyze the provided metrics and the sample of highly negative feedback. "
            "Your goal is to identify the **top 3 underlying causes (the 'Root Causes')** for the poor performance. "
            "Provide the analysis in a formal, structured Markdown report. "
            "The report must contain: 'Overall Diagnosis', 'Worst Performer Focus', 'Top 3 Root Causes (with justification)', and 'Immediate Mitigation Strategy'."
        )

        user_query = (
            f"High-Level Metrics:\n"
            f"- Total Conversations: {total_conversations}\n"
            f"- Net Sentiment Score: {net_sentiment_score:.3f}\n"
            f"- Worst Performing Airline: {worst_airline}\n\n"
            f"Sample of Highly Negative Feedback for Root Cause Identification:\n"
            f"{negative_sample}\n\n"
            "Generate a comprehensive Root Cause Analysis (RCA) report."
        )

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "tools": [{"google_search": {}}], 
        }

        response = self._execute_api_call(payload)
        return _extract_text_from_response(response) if response else "Error in RCA generation."
    
    
    # --- Single Text Analyzer Methods (REQUIRED by 2_Single_Text_Analyzer.py) ---
    def analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single piece of text for sentiment, tone, themes, and actionable insights, 
        returning a structured JSON object.
        """
        system_prompt = (
            "You are an expert conversational AI designed to analyze customer feedback. "
            "Analyze the provided text and provide a comprehensive analysis in the exact JSON format specified in the schema. "
            "Ensure the sentiment score is between -1.0 (very negative) and 1.0 (very positive)."
        )

        user_query = f"Analyze this customer conversation: '{text}'"

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "sentiment_polarity": {"type": "STRING", "description": "Overall sentiment (Positive, Negative, Mixed, Neutral)."},
                "sentiment_score": {"type": "NUMBER", "description": "Numerical score from -1.0 to 1.0."},
                "emotional_tone": {"type": "STRING", "description": "The primary emotion (e.g., Frustration, Delight, Confusion, Curiosity)."},
                "main_themes": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of 3-5 main topics or keywords."},
                "action_type": {"type": "STRING", "description": "The recommended next step (e.g., Escalate, Service Recovery, Product Improvement, No Immediate Action)."},
                "justification": {"type": "STRING", "description": "A brief explanation for the recommended action."}
            },
            "required": ["sentiment_polarity", "sentiment_score", "emotional_tone", "main_themes", "action_type", "justification"],
            "propertyOrdering": ["sentiment_polarity", "sentiment_score", "emotional_tone", "main_themes", "action_type", "justification"]
        }

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
        
        response = self._execute_api_call(payload)
        
        if response:
            try:
                json_string = response['candidates'][0]['content']['parts'][0]['text']
                return json.loads(json_string)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                # Return a dictionary with an error message, as expected by the caller's logic
                return {"error": f"Failed to parse JSON response: {e}. Raw response might be in an incorrect format."}
        else:
            return {"error": "API call failed or returned an empty response."}

    # --- Sentiment Flow Analyzer Method (REQUIRED by 2_Single_Text_Analyzer.py) ---
    def analyze_sentiment_flow(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Analyzes the sentiment of each sentence in a text, returning a list of structured 
        JSON objects for visualization.
        """
        system_prompt = (
            "You are an expert data generator for sentiment visualization. "
            "Your task is to break the provided text into individual sentences and assign a numerical sentiment score "
            "to each sentence. The score must be between -1.0 (most negative) and 1.0 (most positive). "
            "Return the result as a strict JSON array matching the schema."
        )

        user_query = f"Analyze the sentiment flow for each sentence in this text: '{text}'"

        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "sentence_number": {"type": "INTEGER", "description": "The sequential number of the sentence (starting from 1)."},
                    "sentiment_score": {"type": "NUMBER", "description": "The sentiment score of this sentence (-1.0 to 1.0)."},
                },
                "required": ["sentence_number", "sentiment_score"],
                "propertyOrdering": ["sentence_number", "sentiment_score"]
            }
        }

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
        
        response = self._execute_api_call(payload)
        
        if response:
            try:
                json_string = response['candidates'][0]['content']['parts'][0]['text']
                # The response is an array of objects
                return json.loads(json_string)
            except (KeyError, IndexError, json.JSONDecodeError):
                # Return None if parsing fails, which is handled by the caller
                return None
        else:
            return None
