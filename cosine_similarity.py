import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# === CONFIG ===
EMBEDDINGS_PATH = "models/song_embeddings.npy"
METADATA_PATH = "models/metadata.json"

# === LOAD DATA ===
embeddings = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

def get_top_matches(input_embedding, top_n=5):
    """
    Compare input embedding with dataset and return top-N closest songs.
    """
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "title": metadata[idx]["title"],
            "artist": metadata[idx]["artist"],
            "filename": metadata[idx]["filename"],
            "similarity": round(float(similarities[idx]) * 100, 2)
        })

    return results

# === TEST (Optional) ===
if __name__ == "__main__":
    # Dummy test: input = one of the dataset embeddings
    example_input = embeddings[0]
    top_results = get_top_matches(example_input)
    for r in top_results:
        print(r)
