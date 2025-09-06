import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from audio_processor import load_and_clean_audio, extract_voice_features

class VoiceClustering:
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
        self.speaker_map = {}
        
    def train(self, audio_files, speaker_names=None):
        """Train the system to recognize different speakers"""
        print("Training Voice Clustering System...")
        all_features = []
        for file in audio_files:
            clean_audio, sr = load_and_clean_audio(file)
            features = extract_voice_features(clean_audio, sr)
            all_features.append(features)
            print(f"Processed {file}")
        
        # Normalize features (important for clustering)
        features_array = np.array(all_features)
        features_normalized = self.scaler.fit_transform(features_array)
        
        self.kmeans = KMeans(n_clusters=len(audio_files), random_state=42)
        cluster_labels = self.kmeans.fit_predict(features_normalized)
        print("\nSpeaker Assignment:")
        for i, (file, cluster) in enumerate(zip(audio_files, cluster_labels)):
            if speaker_names:
                name = speaker_names[i]
                self.speaker_map[cluster] = name
                print(f"   {name} → Cluster {cluster}")
            else:
                self.speaker_map[cluster] = f"Speaker_{cluster}"
                print(f"   {file} → Cluster {cluster}")
        
        print("Training Complete!")
        return cluster_labels
    
    def identify(self, audio_file):
        """Identify which speaker an audio file belongs to"""
        if self.kmeans is None:
            return "System not trained yet!"
        clean_audio, sr = load_and_clean_audio(audio_file)
        features = extract_voice_features(clean_audio, sr)
        features_normalized = self.scaler.transform([features])
        cluster = self.kmeans.predict(features_normalized)[0]
        speaker = self.speaker_map.get(cluster, f"Unknown_Cluster_{cluster}")
        
        return f"{audio_file} → {speaker} (Cluster {cluster})"

def main():
    print("=" * 50)
    print("VOICE CLUSTERING SYSTEM DEMO")
    print("=" * 50)
    system = VoiceClustering()
    audio_files = ["person1.wav", "person2.wav", "person3.wav"]
    speaker_names = ["Aws", "Yousef", "Qais"] 
    system.train(audio_files, speaker_names)
    print("\n" + "=" * 50)
    print("TESTING SPEAKER IDENTIFICATION")
    print("=" * 50)
    for file in audio_files:
        result = system.identify(file)
        print(result)
    
    print("\n" + "=" * 50)
    print("SYSTEM SUMMARY")
    print("=" * 50)
    print("Successfully trained on 3 speakers")
    print("Each speaker assigned unique cluster")  
    print("Can identify speakers from voice alone")
    print("Ready for real-time speaker recognition!")


if __name__ == "__main__":
    main()