# utils/audio_utils.py

import soundfile as sf


def get_audio_duration(path) -> float:
    audio, sr = sf.read(path)
    return len(audio) / sr