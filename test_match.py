from process_audio import get_embedding
from similarity import get_top_matches

embedding = get_embedding("your_hum_or_clip.wav")
results = get_top_matches(embedding)

for r in results:
    print(r)
