import numpy as np
import librosa
from typing import Dict, Any

def estimate_key(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Estimate musical key using combined chromagram analysis.
    """
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    modes = ['major', 'minor']
    
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    chroma_cqt = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate, hop_length=512, n_chroma=12, bins_per_octave=36)
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=12, hop_length=512)
    y_harmonic, _ = librosa.effects.hpss(audio_data)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harmonic, sr=sample_rate, hop_length=512, n_chroma=12)
    
    combined_chroma = (0.4 * chroma_cqt + 0.2 * chroma_stft + 0.4 * chroma_harm)
    chroma_avg = np.mean(combined_chroma, axis=1)
    
    correlations = []
    for mode_idx, profile in enumerate([major_profile, minor_profile]):
        for key_idx in range(12):
            rolled_profile = np.roll(profile, key_idx)
            correlation = np.corrcoef(rolled_profile, chroma_avg)[0,1]
            correlations.append({
                'key': keys[key_idx],
                'mode': modes[mode_idx],
                'correlation': correlation
            })
    
    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    best_match = correlations[0]
    key_name = f"{best_match['key']} {best_match['mode']}"
    
    confidence = max(0.0, min(1.0, (best_match['correlation'] - correlations[1]['correlation']) * 5))
    
    threshold = best_match['correlation'] * 0.8
    alternatives = [
        f"{c['key']} {c['mode']}"
        for c in correlations[1:4]
        if c['correlation'] > threshold
    ]
    
    return {
        "key": key_name,
        "confidence": float(confidence),
        "secondary_candidates": alternatives,
        "correlation_score": float(best_match['correlation'])
    }
