import numpy as np
import librosa
from typing import Dict

def estimate_tempo(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Estimate the tempo (BPM) and confidence level of an audio signal.
    """
    # Compute the onset envelope with logarithmic scaling
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    
    # Estimate tempo with median aggregation for better accuracy
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate, aggregate=np.median)
    
    # Normalize onset strength to determine confidence
    onset_env_max = np.max(onset_env)
    confidence = np.clip(np.mean(onset_env) / (onset_env_max + 1e-6), 0.0, 1.0)
    
    return {
        "bpm": float(tempo[0]),
        "confidence": float(confidence)
    }
