import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

AUDIO_DIR = "training/dataset/final_wav"
OUTPUT_DIR = "training/dataset/embeddings_ecapa"

os.makedirs(OUTPUT_DIR, exist_ok=True)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa"
)

files = os.listdir(AUDIO_DIR)

for file in tqdm(files):

    if not file.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_DIR, file)

    signal, fs = torchaudio.load(path)

    embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().detach().numpy()

    out_path = os.path.join(
        OUTPUT_DIR,
        file.replace(".wav", ".npy")
    )

    torch.save(embedding, out_path)