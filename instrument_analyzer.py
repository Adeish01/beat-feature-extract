import numpy as np
import librosa
from typing import Dict, Any

def detect_instruments(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Detect instruments based on spectral features.
    """
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
    
    instruments = []
    
    mean_centroid = np.mean(spectral_centroids)
    mean_rolloff = np.mean(spectral_rolloff)
    mean_bandwidth = np.mean(spectral_bandwidth)
    
    if mean_centroid > 3000 and mean_rolloff > 7000:
        instruments.append("High-frequency instruments (possibly cymbals/hi-hats)")
    if mean_centroid < 2000 and mean_bandwidth < 2000:
        instruments.append("Low-frequency instruments (possibly bass)")
    if 200 < mean_centroid < 2000 and mean_bandwidth > 1500:
        instruments.append("Mid-range instruments (possibly guitar/piano)")
    
    return {
        "detected_instruments": instruments,
        "spectral_features": {
            "mean_centroid": float(mean_centroid),
            "mean_rolloff": float(mean_rolloff),
            "mean_bandwidth": float(mean_bandwidth)
        }
    }