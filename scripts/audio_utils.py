import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    """
    Returns duration of audio file in seconds.
    Supports WAV / FLAC / AIFF.
    """
    try:
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
    except Exception as e:
        raise RuntimeError(f"Failed to read audio duration: {e}")