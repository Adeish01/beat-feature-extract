import os
import json
from typing import Dict, Any
import openai
from models.perplexity import get_perplexity_analysis
from models.openai_model import get_openai_analysis


def get_openai_analysis_llm(prompt: str) -> Dict[str, Any]:
    """Get music analysis from OpenAI"""
    return get_openai_analysis(prompt)

def get_perplexity_analysis_llm(prompt: str) -> Dict[str, Any]:
    response =  get_perplexity_analysis(prompt)
    print("Results for perplexity on llm analyzer", response)
    return response

def get_empty_analysis() -> Dict[str, Any]:
    """Return empty analysis structure"""
    return {
        "key": "",
        "tempo": 0.0,
        "mood": [],
        "instruments": [],
        "confidence_scores": {
            "key": 0.0,
            "tempo": 0.0,
            "mood": 0.0,
            "instruments": 0.0
        }
    }
