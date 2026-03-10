import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

audio_dir = "training/dataset/final_wav"
out_dir = "training/dataset/hubert_embeddings"

os.makedirs(out_dir, exist_ok=True)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-large-ls960-ft"
)

model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft"
)

model.eval()

files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

for f in tqdm(files):

    path = os.path.join(audio_dir, f)

    audio, sr = librosa.load(path, sr=16000)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    np.save(
        os.path.join(out_dir, f.replace(".wav",".npy")),
        embedding
    )