import librosa
import numpy as np
from audio_processor import load_and_clean_audio, extract_voice_features

def segment_audio(audio, sr, segment_length=30):
    """Split audio into 30-second segments"""
    samples_per_segment = segment_length * sr
    segments = []
    
    for i in range(0, len(audio), samples_per_segment):
        segment = audio[i:i + samples_per_segment]
        if len(segment) >= samples_per_segment * 0.5:  # Keep segments with 50%+ data
            segments.append(segment)
    
    return segments

def process_all_segments():
    people = ["person1.wav", "person2.wav", "person3.wav"]
    all_segments = []
    segment_labels = []
    
    for person_id, file in enumerate(people):
        audio, sr = load_and_clean_audio(file)
        segments = segment_audio(audio, sr)
        
        for segment in segments:
            features = extract_voice_features(segment, sr)
            all_segments.append(features)
            segment_labels.append(person_id)
        
        print(f"{file}: {len(segments)} segments")
    
    return np.array(all_segments), segment_labels