from pydantic import BaseModel
from typing import Dict, Any
import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIAnalysis(BaseModel):
    musical_key: str
    tempo_bpm: int
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

def get_openai_analysis(prompt: str) -> Dict[str, Any]:
    """Get music analysis from OpenAI with structured output"""
    system_prompt = """Analyze this song and provide:
    - Musical key
    - Tempo in BPM
    - Mood descriptors
    - Instruments detected
    Include confidence scores (0.0-1.0) for each parameter. """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "process_music_analysis",
                "description": "Process the music analysis results",
                "parameters": OpenAIAnalysis.model_json_schema()
            }],
            function_call={"name": "process_music_analysis"},
            temperature=0.7
         )
        
        return json.loads(response.choices[0].message.function_call.arguments)
    except Exception as e:
        print(f"Error in OpenAI analysis: {e}")
        return get_empty_analysis()
