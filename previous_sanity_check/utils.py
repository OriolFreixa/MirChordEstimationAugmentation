import os
import librosa
import json
import pandas as pd
from pathlib import Path
from ACE.inference import run_inference
import previous_sanity_check.auxiliary as a

# PART 1
# Data Loading
def load_chord_data(folder):
    """
    Load chord annotations from JAMS files in the specified folder into a DataFrame.
    Should be in the format:
        file | corpus | chord
    Skip "N" (no chord) entries.
    Params:
        folder (str): Path to the folder containing JAMS files.
    Returns:
        pd.DataFrame: DataFrame with columns 'file', 'corpus', and 'chord'.
    """
    frames = []
    for jams_file in Path(folder).glob("*.jams"):
        j = json.load(open(jams_file))
        file = Path(jams_file).name
        for ann in j["annotations"]:
            if ann.get("namespace") != "chord":
                continue
            corpus = ann["annotation_metadata"]["corpus"]
            for obs in ann["data"]:
                chord = obs["value"]
                if chord != "N":  # Skip no-chord entries
                    frames.append({"file": file, "corpus": corpus, "chord": chord})
    return pd.DataFrame(frames)


# Chord Classification
def classify_chord_harte(chord_str):
    """
    Classify a chord string using Harte notation.

    The function must return a tuple of three booleans:

        (is_major, is_minor, is_other)

    Where exactly one element should be True for valid, parseable chords.
    If the chord cannot be parsed, return (False, False, False).

    You should use `open_harte(chord_str)` from auxiliary.py to obtain the chord object.
    Params:
        chord_str (str): The chord string in Harte notation.
    Returns:
        Tuple[bool, bool, bool]: A tuple indicating if the chord is major, minor, or other.
    """
    # YOUR CODE HERE
    try:
        harte = a.open_harte(chord_str)
        if harte is None:
            return (False, False, False)
    except Exception:
        return (False, False, False)

    return (harte.quality == "major",
            harte.quality == "minor", 
            harte.quality != "major" and harte.quality != "minor")

# Statistics Computation
def get_chord_stats(df, corpora=None):
    """
    Returns a dictionary of chord statistics.
    Params:
        df (pd.DataFrame): DataFrame with chord annotations (from load_chord_data).
        corpora (List[str] or None): List of corpus names to filter by. If None, use all.
    Returns:
        Dict[str, int]: Dictionary with keys 'total_chords', 'unique_chords',
                        'major', 'minor', 'other'.
    """
    if (corpora is not None) and len(corpora) > 0:
        df = df[df['corpus'].isin(corpora)]
    total_chords = len(df)
    unique_chords = df['chord'].nunique()
    major = minor = other = 0
    for chord in df['chord']:
        is_major, is_minor, is_other = classify_chord_harte(chord)
        if is_major:
            major += 1
        elif is_minor:
            minor += 1
        elif is_other:
            other += 1
    return {
        'total_chords': total_chords,
        'unique_chords': unique_chords,
        'major': major,
        'minor': minor,
        'other': other
    }

# Chroma Feature Computation
def compute_chroma_features(audio_path, duration=10, sr=22050, hop_length=2048):
    """
    Using librosa load an audio file and compute STFT chroma and CENS chroma features.
    It should return a dictionary that can be ingested by the plot_chroma_features() function in auxiliary.py.

    Params:
        audio_path (str): Path to the audio file.
        duration (float): Duration in seconds to load from the audio file.
        sr (int): Sampling rate for loading the audio.
        hop_length (int): Hop length for chroma feature computation.
    Returns:
        Dict[str, Any]: Dictionary with keys 'chroma_stft', 'chroma_cens', 'sr', 'hop_length'.
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    stftchroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    censchroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, win_len_smooth=1)
    return {
        "chroma_stft": stftchroma,
        "chroma_cens": censchroma,
        "sr": sr,
        "hop_length": hop_length
    }

# PART 2
# When comparing conformer vs conformer_decomposed models
def run_inference_on_folder(model_name, audio_folder, checkpoint, vocab_path, output_folder):
    """
    Run chord recognition inference on all audio files in a folder. 
    Use from ACE.inference import run_inference
    Params:
        model_name (str): Name of the model to use ('conformer' or 'conformer_decomposed').
        audio_folder (str): Path to the folder containing audio files.
        checkpoint (str): Path to the model checkpoint file.
        vocab_path (str): Path to the vocabulary file.
        output_folder (str): Path to the folder where output .lab files will be saved.
    Returns:
        None
    """
    for audio in Path(audio_folder).glob("*.mp3"):
        out_lab = Path(output_folder) / (audio.stem + "_model.lab")

        run_inference(
            audio_path=audio, 
            checkpoint=checkpoint,
            vocab_path=vocab_path, 
            out_lab=out_lab,
            model_name=model_name,
            chord_min_duration=0.5
            )

