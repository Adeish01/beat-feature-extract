import numpy as np
import librosa
from typing import Dict, Any

def analyze_mood(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Analyze the mood of the audio based on various audio features.
    """
    # Extract relevant features for mood analysis
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    tempo = librosa.beat.tempo(y=audio_data, sr=sample_rate)[0]
    
    # Calculate energy
    rms = librosa.feature.rms(y=audio_data)[0]
    energy = np.mean(rms)
    
    # Calculate brightness
    brightness = np.mean(spectral_centroids)
    
    # Determine moods based on audio features
    moods = []
    confidence = 0.0
    
    # Energy-based mood detection
    if energy > 0.1:
        if tempo > 120:
            moods.append("Energetic")
            confidence += 0.3
        else:
            moods.append("Calm")
            confidence += 0.2
    
    # Brightness-based mood detection
    if brightness > 2000:
        moods.append("Bright")
        confidence += 0.2
    else:
        moods.append("Dark")
        confidence += 0.2
    
    # Normalize confidence
    confidence = min(confidence, 1.0)
    
    return {
        "moods": moods,
        "confidence": confidence,
        "features": {
            "energy": float(energy),
            "brightness": float(brightness),
            "tempo": float(tempo)
        }
    }