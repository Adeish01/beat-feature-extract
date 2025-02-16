import io
import librosa
import streamlit as st
from typing import Dict, Any
import numpy as np
from key_analyzer import estimate_key
from tempo_analyzer import estimate_tempo
from instrument_analyzer import detect_instruments
from mood_analyzer import analyze_mood 
from llm_analyzers import get_perplexity_analysis_llm



def get_standardized_output() -> Dict[str, Any]:
    """
    Standard output format for all analysis types
    """
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
def process_audio_file(file_content):
    """Process audio file content and return audio data with sample rate"""
    audio_bytes = io.BytesIO(file_content)
    audio_data, sample_rate = librosa.load(audio_bytes)
    return audio_data, sample_rate

# def download_youtube_audio(url: str):
#     """Download audio from YouTube URL using youtube-dl"""
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)  # Changed to download=True
#         filename = ydl.prepare_filename(info)
#         # Load the downloaded file directly
#         audio_data, sample_rate = librosa.load(filename, sr=None)
#         return audio_data, sample_rate, info['title']

def analyze_audio_data(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Common analysis function for all audio data
    """
    output = get_standardized_output()
    
    if audio_data is not None and sample_rate is not None:
        # Analyze key
        key_info = estimate_key(audio_data, sample_rate)
        output["key"] = key_info["key"]
        output["confidence_scores"]["key"] = key_info["confidence"]
        
        # Analyze tempo
        tempo_info = estimate_tempo(audio_data, sample_rate)
        if "tempo" in output:
            output["tempo"] = tempo_info["bpm"]
            output["confidence_scores"]["tempo"] = tempo_info["confidence"]
        else:
            output["bpm"] = tempo_info["bpm"]
            output["confidence_scores"]["bpm"] = tempo_info["confidence"]
        
        # Analyze instruments
        instrument_info = detect_instruments(audio_data, sample_rate)
        output["instruments"] = instrument_info["detected_instruments"]
        
        # Analyze mood based on audio features
        mood_info = analyze_mood(audio_data, sample_rate)
        output["mood"] = mood_info["moods"]
        output["confidence_scores"]["mood"] = mood_info["confidence"]
    
    return output


def analyze_text_with_llms(prompt: str) -> Dict[str, Any]:
    """
    Function to analyze text using multiple LLM models
    """
    # Get analysis from Perplexity
    perplexity_result = get_perplexity_analysis_llm(prompt)
    
    # Since perplexity_result is already a dictionary, return it directly
    return perplexity_result



def analyze_audio_file(uploaded_file) -> Dict[str, Any]:
    """
    Function to analyze uploaded audio file
    """
    try:
        audio_data, sample_rate = process_audio_file(uploaded_file.read())
        return analyze_audio_data(audio_data, sample_rate)
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return get_standardized_output()

# def analyze_url_audio(url: str) -> Dict[str, Any]:
#     """
#     Function to analyze audio from URL
#     """
#     try:
#         audio_data, sample_rate, _ = download_youtube_audio(url)
#         return analyze_audio_data(audio_data, sample_rate)
#     except Exception as e:
#         st.error(f"Error processing URL audio: {str(e)}")
#         return get_standardized_output()

def analyze_system_recording() -> Dict[str, Any]:
    """
    Function to analyze system recording
    """
    # This would be implemented with actual recording functionality
    output = get_standardized_output()
    # Placeholder for recording implementation
    return output

def display_analysis_results(results: Dict[str, Any]):
    """
    Standardized display of analysis results
    """
    col1, col2 = st.columns(2)
    bpm = "tempo"
    if "tempo" in results:
        bpm = "tempo"
    else:
        bpm = "bpm"
    
    with col1:
        st.metric(
            "Key", 
            results["key"],
            help=f"Confidence: {results['confidence_scores']['key']:.1%}"
        )
        st.metric(
            "Tempo", 
            f"{results[bpm]:.1f} BPM",
            help=f"Confidence: {results['confidence_scores'][bpm]:.1%}"
        )
    
    with col2:
        st.subheader("Detected Mood")
        if results["mood"]:
            for mood in results["mood"]:
                st.write(f"• {mood}")
        else:
            st.write("No mood detected")
            
        st.subheader("Detected Instruments")
        if results["instruments"]:
            for instrument in results["instruments"]:
                st.write(f"• {instrument}")
        else:
            st.write("No instruments detected")
def main():
    st.title("🎯 Music Analysis Hub")
    
    tab1, tab2, = st.tabs([
        "LLM Analysis", 
        "Audio File Analysis (Librosa)",
        # "URL Audio Analysis (librosa)",
        # "Live Recording Analysis (librosa)"
    ])
    
    with tab1:
        st.header("Multi-LLM Analysis")
        prompt = st.text_area("Enter your prompt:")
        if st.button("Analyze Text"):
            with st.spinner("Processing with LLMs..."):
                results = analyze_text_with_llms(prompt)
                display_analysis_results(results)
    
    with tab2:
        st.header("Audio File Analysis")
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "ogg", "flac"]
        )
        if uploaded_file:
            st.audio(uploaded_file)
            with st.spinner("Analyzing audio..."):
                results = analyze_audio_file(uploaded_file)
                display_analysis_results(results)
    
    # with tab3:
    #     st.header("URL Audio Analysis")
    #     url = st.text_input("Enter audio URL:")
    #     if st.button("Analyze URL"):
    #         with st.spinner("Downloading and analyzing..."):
    #             results = analyze_url_audio(url)
    #             display_analysis_results(results)
    
    # with tab4:
    #     st.header("Live Recording Analysis")
    #     if st.button("Start Recording"):
    #         with st.spinner("Recording and analyzing..."):
    #             results = analyze_system_recording()
    #             display_analysis_results(results)
    
    with st.expander("Show Raw JSON Output"):
        if 'results' in locals():
            st.json(results)

if __name__ == "__main__":
    main()
