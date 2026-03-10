import torch
import librosa
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from speechbrain.pretrained import EncoderClassifier
from TTS.api import TTS

# -------------------------
# Age model architecture
# -------------------------

class FinalAgeModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.age_embed = torch.nn.Embedding(8, 32)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(1216 + 32, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1216),
        )

    def forward(self, x, age):
        a = self.age_embed(age)
        z = torch.cat([x, a], dim=1)
        return self.net(z)


# -------------------------
# Age group mapping
# -------------------------

age_map = {
    "5-10": 0,
    "10-15": 1,
    "15-20": 2,
    "20-30": 3,
    "30-40": 4,
    "40-50": 5,
    "50-60": 6,
    "60+": 7,
}

# -------------------------
# Load models
# -------------------------

print("Loading models...")

hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-large-ls960-ft"
)

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

age_model = FinalAgeModel()
age_model.load_state_dict(torch.load("training/models/final_age_model.pt"))
age_model.eval()

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

print("Models loaded")


# -------------------------
# Feature extraction
# -------------------------

def extract_features(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    # ECAPA embedding
    signal = torch.tensor(audio).unsqueeze(0)
    ecapa_emb = ecapa.encode_batch(signal).squeeze().detach().numpy()

    # HuBERT embedding
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        out = hubert(**inputs)

    hubert_emb = out.last_hidden_state.mean(dim=1).squeeze().numpy()

    combined = np.concatenate([ecapa_emb, hubert_emb])

    return combined


# -------------------------
# Age transformation
# -------------------------

def transform_voice(audio_path, target_age_group):

    emb = extract_features(audio_path)

    emb = torch.tensor(emb).unsqueeze(0).float()

    age_index = age_map[target_age_group]
    age = torch.tensor([age_index])

    with torch.no_grad():
        new_emb = age_model(emb, age)

    return new_emb.squeeze().numpy()


# -------------------------
# Example run
# -------------------------

input_audio = "input_voice.wav"

print("Transforming voice...")

new_embedding = transform_voice(input_audio, "60+")

print("Generating aged voice...")

tts.tts_to_file(
    text="Hello this is how your voice may sound when you are older.",
    speaker_wav=input_audio,
    language="en",
    file_path="aged_voice.wav",
)

print("Generated aged_voice.wav")