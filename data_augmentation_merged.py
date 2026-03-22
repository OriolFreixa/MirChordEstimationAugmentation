"""
data_augmentation.py
====================
Unified augmentation functions for the Conformer-based ACE project.
All functions share the same signature:

    augment(input_file: str, output_file: str, args: dict) -> str

The returned string is the final output path (always .mp3).

Wet/dry defaults are set to 0.5 (50 % wet / 50 % dry) for reverb and
real-noise augmentations, following the project convention.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, convolve, sosfilt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_mp3_path(output_file: str) -> str:
    """Return *output_file* with its extension replaced by .mp3."""
    base, _ = os.path.splitext(output_file)
    return f"{base}.mp3"


def _write_mp3(output_file: str, audio: np.ndarray, sr: int) -> str:
    """Write *audio* as MP3 and return the final path used."""
    output_mp3 = _as_mp3_path(output_file)
    sf.write(output_mp3, audio, sr, format="MP3")
    return output_mp3


def _load_audio(input_file: str, sr: int) -> tuple[np.ndarray, int]:
    """Load an audio file with librosa and return (samples, sample_rate)."""
    return librosa.load(input_file, sr=sr)


# ---------------------------------------------------------------------------
# 1. Convolutional reverb  (IR-based)
# ---------------------------------------------------------------------------

def apply_reverb(input_file: str, output_file: str, args: dict) -> str:
    """Apply convolutional reverb using an impulse response (IR) file.

    Args:
        input_file: Path to the source audio file.
        output_file: Desired output path (extension is forced to .mp3).
        args:
            ir_path   (str)   – Path to the impulse-response audio file. Required.
            sr        (int)   – Target sample rate. Default: 22 050.
            wet_dry_mix (float) – 0.0 = fully dry, 1.0 = fully wet. Default: 0.5.
    """
    ir_path = args.get("ir_path")
    if ir_path is None:
        raise ValueError("apply_reverb: 'ir_path' must be provided in args.")

    sr = args.get("sr", 22050)
    wet_dry_mix = args.get("wet_dry_mix", 0.5)

    y, sr_loaded = _load_audio(input_file, sr)
    ir, _ = librosa.load(ir_path, sr=sr_loaded)

    y_wet = convolve(y, ir, mode="full")[: len(y)]
    y_wet = librosa.util.normalize(y_wet)

    y_out = (1.0 - wet_dry_mix) * y + wet_dry_mix * y_wet
    y_out = librosa.util.normalize(y_out)

    return _write_mp3(output_file, y_out, sr_loaded)


# ---------------------------------------------------------------------------
# 2. Real-noise addition  (background noise from a WAV/MP3 file)
# ---------------------------------------------------------------------------

def add_real_noise(input_file: str, output_file: str, args: dict) -> str:
    """Mix a real-world background-noise recording into the source audio.

    The noise clip is tiled or trimmed to match the source length before mixing.

    Args:
        input_file: Path to the source audio file.
        output_file: Desired output path (extension is forced to .mp3).
        args:
            noise_path  (str)   – Path to the background-noise audio file. Required.
            sr          (int)   – Target sample rate. Default: 22 050.
            wet_dry_mix (float) – Noise level relative to source.
                                  0.0 = no noise, 1.0 = noise only. Default: 0.5.
    """
    noise_path = args.get("noise_path")
    if noise_path is None:
        raise ValueError("add_real_noise: 'noise_path' must be provided in args.")

    sr = args.get("sr", 22050)
    wet_dry_mix = args.get("wet_dry_mix", 0.5)

    y, sr_loaded = _load_audio(input_file, sr)
    noise_raw, _ = librosa.load(noise_path, sr=sr_loaded)

    # Tile or trim noise to match source length
    if len(noise_raw) < len(y):
        repeats = int(np.ceil(len(y) / len(noise_raw)))
        noise_raw = np.tile(noise_raw, repeats)
    noise_raw = noise_raw[: len(y)]

    # Normalise noise to unit RMS, then scale to target mix level
    rms_noise = np.sqrt(np.mean(noise_raw ** 2)) + 1e-12
    noise_normalised = noise_raw / rms_noise

    rms_signal = np.sqrt(np.mean(y ** 2)) + 1e-12
    noise_scaled = noise_normalised * rms_signal * wet_dry_mix

    y_out = (1.0 - wet_dry_mix) * y + noise_scaled
    peak = np.max(np.abs(y_out))
    if peak > 1.0:
        y_out = y_out / peak

    return _write_mp3(output_file, y_out, sr_loaded)


# ---------------------------------------------------------------------------
# 3. Synthetic noise addition  (white / pink / brown)
# ---------------------------------------------------------------------------

def add_noise(input_file: str, output_file: str, args: dict) -> str:
    """Add synthetically generated noise to the source audio.

    Args:
        input_file: Path to the source audio file.
        output_file: Desired output path (extension is forced to .mp3).
        args:
            sr         (int)   – Target sample rate. Default: 22 050.
            noise_type (str)   – 'white', 'pink', or 'brown'. Default: 'white'.
            snr_db     (float) – Signal-to-noise ratio in dB. Default: 25.
    """
    sr = args.get("sr", 22050)
    noise_type = args.get("noise_type", "white")
    snr_db = float(args.get("snr_db", 25))

    y, sr_loaded = _load_audio(input_file, sr)

    signal_power = np.mean(y ** 2)
    if signal_power <= 1e-12:
        return _write_mp3(output_file, y, sr_loaded)

    noise_power = signal_power / (10 ** (abs(snr_db) / 10))

    if noise_type == "white":
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    elif noise_type == "pink":
        white = np.random.normal(0, 1, len(y))
        noise = np.convolve(white, [1, 1], mode="same")
        noise = noise / (np.std(noise) + 1e-12) * np.sqrt(noise_power)
    elif noise_type == "brown":
        white = np.random.normal(0, 1, len(y))
        noise = np.cumsum(white)
        noise = noise / (np.std(noise) + 1e-12) * np.sqrt(noise_power)
    else:
        raise ValueError(f"add_noise: unknown noise_type '{noise_type}'.")

    y_out = y + noise
    peak = np.max(np.abs(y_out))
    if peak > 1.0:
        y_out = y_out / peak

    return _write_mp3(output_file, y_out, sr_loaded)


# ---------------------------------------------------------------------------
# 4. Random equalisation
# ---------------------------------------------------------------------------

def randomly_eq(input_file: str, output_file: str, args: dict) -> str:
    """Apply random per-band gain to simulate different playback or recording EQ.

    Args:
        input_file: Path to the source audio file.
        output_file: Desired output path (extension is forced to .mp3).
        args:
            sr         (int)          – Target sample rate. Default: 22 050.
            bands      (list[tuple])  – Frequency bands [(low_hz, high_hz), ...].
                                        Default: [(100, 500), (500, 2000), (2000, 8000)].
            gain_range (tuple[float]) – (min_dB, max_dB) per band. Default: (-6, 6).
    """
    sr = args.get("sr", 22050)
    default_bands = [(100, 500), (500, 2000), (2000, 8000)]
    bands = args.get("bands", default_bands)
    gain_range = args.get("gain_range", (-6, 6))

    y, sr_loaded = _load_audio(input_file, sr)
    nyquist = sr_loaded / 2.0

    # Clamp bands to valid range
    valid_bands = []
    for low, high in bands:
        lo = max(1.0, min(float(low), nyquist - 1))
        hi = min(float(high), nyquist - 1)
        if lo < hi:
            valid_bands.append((lo, hi))

    if not valid_bands:
        valid_bands = [
            (100.0, nyquist * 0.1),
            (nyquist * 0.1, nyquist * 0.4),
            (nyquist * 0.4, nyquist * 0.8),
        ]

    y_eq = np.zeros_like(y)
    for low, high in valid_bands:
        sos = butter(2, [low, high], btype="bandpass", fs=sr_loaded, output="sos")
        y_band = sosfilt(sos, y)
        gain_db = np.random.uniform(gain_range[0], gain_range[1])
        y_band *= 10 ** (gain_db / 20.0)
        y_eq += y_band

    y_eq = librosa.util.normalize(y_eq)
    return _write_mp3(output_file, y_eq, sr_loaded)


# ---------------------------------------------------------------------------
# 5. Dynamic compression
# ---------------------------------------------------------------------------

def apply_compression(input_file: str, output_file: str, args: dict) -> str:
    """Apply a simple feed-forward dynamic compressor.

    Args:
        input_file: Path to the source audio file.
        output_file: Desired output path (extension is forced to .mp3).
        args:
            sr              (int)   – Target sample rate. Default: 22 050.
            threshold_db    (float) – Compression threshold in dB. Default: -20.
            ratio           (float) – Compression ratio (>1). Default: 4.0.
            attack_ms       (float) – Attack time in ms. Default: 5.
            release_ms      (float) – Release time in ms. Default: 100.
            makeup_gain_db  (float) – Make-up gain in dB. Default: 0.
    """
    sr = args.get("sr", 22050)
    threshold_db = args.get("threshold_db", -20)
    ratio = args.get("ratio", 4.0)
    attack_ms = args.get("attack_ms", 5)
    release_ms = args.get("release_ms", 100)
    makeup_gain_db = args.get("makeup_gain_db", 0)

    y, sr_loaded = _load_audio(input_file, sr)

    threshold_linear = 10 ** (threshold_db / 20.0)
    attack_samples = max(1, int(attack_ms * sr_loaded / 1000))
    release_samples = max(1, int(release_ms * sr_loaded / 1000))
    makeup_gain = 10 ** (makeup_gain_db / 20.0)

    y_compressed = np.zeros_like(y)
    envelope = np.zeros_like(y)

    for i in range(len(y)):
        prev_env = envelope[i - 1] if i > 0 else 0.0
        abs_sample = abs(y[i])
        if abs_sample > prev_env:
            envelope[i] = prev_env + (abs_sample - prev_env) / attack_samples
        else:
            envelope[i] = prev_env * (1.0 - 1.0 / release_samples)

        if envelope[i] > threshold_linear:
            gain_reduction = (envelope[i] / threshold_linear) ** (1.0 / ratio - 1.0)
            gain = 1.0 / gain_reduction
        else:
            gain = 1.0

        y_compressed[i] = y[i] * gain * makeup_gain

    y_compressed = librosa.util.normalize(y_compressed)
    return _write_mp3(output_file, y_compressed, sr_loaded)