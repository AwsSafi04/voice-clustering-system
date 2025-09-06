import librosa
import numpy as np

def load_and_clean_audio(filename):
    """Load audio file and remove silence"""
    audio, sample_rate = librosa.load(filename, sr=16000)
    intervals = librosa.effects.split(audio, top_db=20)
    if len(intervals) > 0:
        clean_audio = np.concatenate([audio[start:end] for start, end in intervals])
    else:
        clean_audio = audio
    return clean_audio, sample_rate

def extract_voice_features(audio, sample_rate):
    """Extract features that represent speaker's voice characteristics"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    features = np.concatenate([
        np.mean(mfcc, axis=1),                    # Average MFCC values
        np.std(mfcc, axis=1),                     # MFCC variation
        np.mean(spectral_centroids, axis=1),      # Average brightness
        np.mean(spectral_rolloff, axis=1)         # Average rolloff
    ])
    return features

people = ["person1.wav", "person2.wav", "person3.wav"]
print("Processing audio files...")
for person in people:
    clean_audio, sr = load_and_clean_audio(person)
    features = extract_voice_features(clean_audio, sr)
    
    original_duration = librosa.get_duration(filename=person)
    clean_duration = len(clean_audio) / sr
    print(f"{person}:")
    print(f"  Original: {original_duration:.1f}s")
    print(f"  After removing silence: {clean_duration:.1f}s") 
    print(f"  Voice features: {len(features)} numbers")
    print()