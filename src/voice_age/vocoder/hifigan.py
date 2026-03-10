import torch
import json
import numpy as np
from scipy.io.wavfile import write

from pathlib import Path

# Import HiFiGAN model
import sys
sys.path.append("hifi-gan")

from models import Generator
from env import AttrDict


class HiFiGANVocoder:

    def __init__(self):

        config_path = "models/hifigan/config.json"
        model_path = "models/hifigan/generator_v3"

        with open(config_path) as f:
            config = AttrDict(json.load(f))

        self.generator = Generator(config)

        checkpoint = torch.load(model_path, map_location="cpu")
        self.generator.load_state_dict(checkpoint["generator"])

        self.generator.eval()
        self.generator.remove_weight_norm()

    def mel_to_audio(self, mel):

        mel = torch.FloatTensor(mel).unsqueeze(0)

        with torch.no_grad():
            audio = self.generator(mel)

        audio = audio.squeeze().cpu().numpy()

        return audio