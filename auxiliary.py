import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

from typing import Union, List, Dict, Tuple

from harte.harte import Harte

import librosa

# Chord Quality Analysis (Harte)
def open_harte(chord_str: str) -> Harte:
    """
    Creates a Harte chord object for the given chord.
    This is an extension of the music21.chord class.

    Args:
        chord_str (str): The chord to create a Harte object for.

    Returns:
        Harte: The Harte object for the given chord.
    """
    if "/bb1" in chord_str:
        chord_str = chord_str.replace("/bb1", "/b7")

    return Harte(chord_str)

def plot_chord_distribution(stats, title = ""):
    labels = ['Major', 'Minor', 'Other']
    sizes = [stats['major'], stats['minor'], stats['other']]
    colors = ['#ff9999','#66b3ff','#99ff99']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plot_chroma_features(chroma_dict):
    """
    Plot STFT chroma and CENS chroma from a chroma feature dictionary.

    Parameters
    ----------
    chroma_dict : dict
        Output from compute_chroma_features().
    """

    chroma_stft = chroma_dict["chroma_stft"]
    chroma_cens = chroma_dict["chroma_cens"]
    sr = chroma_dict["sr"]
    hop_length = chroma_dict["hop_length"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot STFT chroma
    img1 = librosa.display.specshow(
        chroma_stft,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=hop_length,
        cmap="gray_r",        # <--- grayscale
        ax=ax[0]
    )
    ax[0].set_title("Chroma STFT")
    fig.colorbar(img1, ax=ax[0])

    # Plot CENS chroma
    img2 = librosa.display.specshow(
        chroma_cens,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=hop_length,
        cmap="gray_r",        # <--- grayscale
        ax=ax[1]
    )
    ax[1].set_title("Chroma CENS")
    fig.colorbar(img2, ax=ax[1])

    plt.tight_layout()
    plt.show()

def get_jams_for_audio(audio_path, jams_dir="./jams_augment_small"):
	"""Return path to JAMS file corresponding to audio_path."""
	base = os.path.splitext(os.path.basename(audio_path))[0]
	jams_path = os.path.join(jams_dir, base + ".jams")
	if not os.path.isfile(jams_path):
		raise FileNotFoundError(f"JAMS file not found for {audio_path} at {jams_path}")
	return jams_path

def convert_jams_to_lab_file(jams_path, lab_path):
	"""Convert JAMS file to lab file with onsets and offsets."""
	import jams
	j = jams.load(jams_path)
	# Assuming the first annotation contains the desired data
	ann = j.annotations[0]
	with open(lab_path, "w") as f:
		for obs in ann.data:
			start = obs.time
			end = obs.time + obs.duration
			label = obs.value
			f.write(f"{start:.6f}	{end:.6f}	{label}\n")





SR = 22050
HOP = 2048

def load_and_plot_cqt(filepath):
    """
    Load a .pt file containing (cqt_tensor, jams_chunk)
    and plot the CQT using matplotlib, with time on x-axis.
    """
    cqt_tensor, _ = torch.load(filepath)

    # shape (1, freq_bins, frames) → (freq_bins, frames)
    cqt = cqt_tensor.squeeze(0).numpy()
    num_frames = cqt.shape[1]

    # compute time axis
    times = np.arange(num_frames) * (HOP / SR)

    plt.figure(figsize=(10, 4))

    # use extent= to map frames → seconds
    plt.imshow(
        cqt,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, cqt.shape[0]],
        cmap="gray_r"
    )

    plt.title("CQT")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency bins")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()


def load_and_plot_cqt_with_chords(
    pt_file,
    vocab_file="chords_vocab.joblib",
    label_key="complete"
):
    """
    Load a .pt CQT file and its chord labels, then plot the CQT with chord boundaries
    and chord names on a time axis (seconds).
    """

    # -------- Load data --------
    cqt_tensor, jams = torch.load(pt_file)
    encoder = joblib.load(vocab_file)
    chord_names = encoder.classes_

    # CQT shape: (1, freq_bins, n_frames)
    cqt = cqt_tensor.squeeze(0).numpy()
    n_bins, n_frames = cqt.shape

    # Framewise chord labels
    labels = jams[label_key].numpy()

    # -------- Build segment list (start_frame, end_frame, chord_id) --------
    segments = []
    start = 0
    cur = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != cur:
            segments.append((start, i - 1, cur))
            start = i
            cur = labels[i]

    segments.append((start, len(labels) - 1, cur))

    # -------- Time axis conversion --------
    # time_at_frame = frame * (HOP / SR)
    frame_to_sec = HOP / SR
    total_time = (n_frames - 1) * frame_to_sec

    # -------- Plot CQT with time extent --------
    plt.figure(figsize=(14, 6))
    plt.imshow(
        cqt,
        aspect='auto',
        origin='lower',
        extent=[0, total_time, 0, n_bins],   # <-- time axis (seconds)
        cmap="gray_r",
    )
    plt.title("CQT with Chord Boundaries")
    plt.ylabel("Frequency bins")
    plt.xlabel("Time (seconds)")

    # -------- Draw chord segments --------
    for (s, e, lab) in segments:
        chord_str = chord_names[lab] if 0 <= lab < len(chord_names) else "UNK"

        # convert to seconds
        s_t = s * frame_to_sec
        e_t = e * frame_to_sec
        center_t = (s_t + e_t) / 2

        # Vertical boundary at segment start
        plt.axvline(
            x=s_t,
            color='white',
            linestyle='--',
            linewidth=1,
            alpha=0.7
        )

        # Chord label near the top
        plt.text(
            center_t,
            n_bins - 5,
            chord_str,
            ha='center',
            va='top',
            color='yellow',
            fontsize=10,
            weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    plt.tight_layout()
    plt.show()


# part 3: for creating new splits of the dataset
def copy_selected_files(source_folder: str, destination_folder: str, prefixes: List[str]):
    import shutil
    import os

    # ensure destination is a directory
    os.makedirs(destination_folder, exist_ok=True)

    copied = 0
    for filename in os.listdir(source_folder):
        src = os.path.join(source_folder, filename)
        if not os.path.isfile(src):
            continue
        if any(filename.startswith(prefix) for prefix in prefixes):
            dst = os.path.join(destination_folder, filename)
            shutil.copy2(src, dst)  # copy2 preserves metadata
            copied += 1

    print(f"Copied {copied} files from {source_folder} to {destination_folder}")

# # Pretty Printing Helper
def print_chord_stats(stats: Dict, title: str = ""):
    if title:
        print("=" * 60)
        print(title)
        print("=" * 60)

    print(f"Total chords: {stats['total_chords']:,}")
    print(f"Unique chords: {stats['unique_chords']}")
    print()