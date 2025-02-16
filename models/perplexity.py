import json
import requests
import streamlit as st
from pydantic import BaseModel
from typing import Dict, Any



PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

class PerplexityAnalysis(BaseModel):
    musical_key: str | None
    tempo_bpm: int | None
    mood_descriptors: list[str]
    instruments_detected: list[str]
    confidence_scores: Dict[str, float]

def get_empty_analysis() -> Dict[str, Any]:
    return {
        "musical_key": None,
        "tempo_bpm": None,
        "mood_descriptors": [],
        "instruments_detected": [],
        "confidence_scores": {}
    }

def extract_song_name_with_model(user_input: str) -> str:
    """Use an AI model to extract the song name from user input."""
    if not user_input or not user_input.strip():
        print("Error: Empty user input")
        return None
    
    system_prompt = """Extract this exact format given in the examples. 
    If no song name and artise name is not found, return an empty string. 
    If song name is found but artist name is not found, find the artist name and return EXACT output in the examples.   
    Do NOT include artist names, additional text, or extra details.
    
    Example Inputs and Outputs:
    - Input: "Find details for 'Bohemian Rhapsody by Queen'" → Output: " Get me accurate key, bpm, moods, instruments, for Bohemian Rhapsody by Queen"
    - Input: "Get me the BPM of 'Blinding Lights' by The Weeknd" → Output: "Get me accurate key, bpm, moods, instruments, for Blinding Lights by The Weeknd"
    - Input: "Tell me about 'Imagine' by John Lennon" → Output: "Get me accurate key, bpm, moods, instruments, for Imagine by John Lennon"
    """
    
    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            content = response_json["choices"][0].get("message", {}).get("content", "").strip()
            return content  # Expected to be just the song name
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

    return ""  # Default empty if extraction fails


def extract_json_from_text(text: str) -> dict:
    """Extract JSON object from text that might contain additional content."""
    try:
        # Find the first occurrence of '{'
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in text")
            
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # We found the matching closing brace
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
                    
        raise ValueError("No complete JSON object found")
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return get_empty_analysis()

def get_perplexity_analysis(prompt: str) -> Dict[str, Any]:
    """Get music analysis from Perplexity AI with Sonar Pro structured output."""
    user_prompt = extract_song_name_with_model(prompt)
    print(f"User prompt: {user_prompt}")
    if not user_prompt:
        print("Failed to extract valid song information from prompt")
        return get_empty_analysis()
    
    system_prompt = """ABSOLUTELY SEARCH THE WEB to get the best answer.
        Return ONLY a JSON object with these exact keys (no other text):
        {
            "key": "musical key",
            "bpm": number,
            "mood": "comma-separated moods",
            "instruments": "comma-separated instruments",
            "confidence_scores": {
                "key": float 0-1,
                "bpm": float 0-1,
                "mood": float 0-1,
                "instruments": float 0-1
            }
        }"""

    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    # try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    response_json = response.json()
    
    if "choices" in response_json and len(response_json["choices"]) > 0:
        content = response_json["choices"][0].get("message", {}).get("content", "")
        
        if content:
            # Extract and parse the JSON from the content
            parsed_data = extract_json_from_text(content)
            
            respomse_data = {
                "key": parsed_data.get("key"),
                "bpm": parsed_data.get("bpm"),
                "mood": [m.strip() for m in parsed_data.get("mood", "").split(",")],
                "instruments": [i.strip() for i in parsed_data.get("instruments", "").split(",")],
                "confidence_scores": parsed_data.get("confidence_scores", {})
            }
            return respomse_data

