import os
import torch
import numpy as np
import librosa
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

audio_dir = "training/dataset/final_wav"
out_dir = "training/dataset/embeddings"

os.makedirs(out_dir, exist_ok=True)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

for f in tqdm(files):

    path = os.path.join(audio_dir, f)

    signal, sr = librosa.load(path, sr=16000)
    signal = torch.tensor(signal).unsqueeze(0)

    embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().detach().numpy()

    np.save(
        os.path.join(out_dir, f.replace(".wav",".npy")),
        embedding
    )