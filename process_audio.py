import openl3
import librosa

# function get_embedding translates the audio file into a 512-dimensional embedding vector. This will be used tot turn the songs into machine readable vectors as well as user recirdings to match for similarity.
def get_embedding(filepath):
    audio, sr = librosa.load(filepath, sr=48000, mono=True) # loads the audio file from the filepath
    emb, _ = openl3.get_audio_embedding(audio, sr, # passing the audio into OpenL3 to extract the embedding
                                        input_repr="mel256", # uses a mel spectrogram with 256 bins as the audio representation
                                        content_type="music", #tells OpenL3 to use its music trained model.
                                        embedding_size=512) # size of the embedding vector
    return emb.mean(axis=0)
