import numpy as np
from sklearn.cluster import KMeans
from audio_processor import load_and_clean_audio, extract_voice_features

def cluster_voices():
    """Group similar voices together"""
    people = ["person1.wav", "person2.wav", "person3.wav"]
    all_features = []
    print("Extracting voice signatures...")
    for person in people:
        clean_audio, sr = load_and_clean_audio(person)
        features = extract_voice_features(clean_audio, sr)
        all_features.append(features)
        print(f"{person}")
    features_array = np.array(all_features)
    print(f"\nFeature matrix shape: {features_array.shape}")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(features_array)
    print("\n=== Clustering Results ===")
    for i, (person, cluster) in enumerate(zip(people, cluster_labels)):
        print(f"{person} in Cluster {cluster}")
    unique_clusters = len(set(cluster_labels))
    if unique_clusters == 3:
        print("\nEach person got a different cluster")
    else:
        print(f"\nOnly {unique_clusters} unique clusters found")
    
    return cluster_labels, features_array

cluster_voices()