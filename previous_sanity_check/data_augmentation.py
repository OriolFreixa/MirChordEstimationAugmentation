import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, convolve


def _as_mp3_path(output_file: str) -> str:
    """Return output path with .mp3 extension."""
    base, _ = os.path.splitext(output_file)
    return f"{base}.mp3"


def _write_mp3(output_file: str, audio: np.ndarray, sr: int) -> str:
    """Write audio as MP3 and return the final path used."""
    output_mp3 = _as_mp3_path(output_file)
    sf.write(output_mp3, audio, sr, format="MP3")
    return output_mp3

def augment_data(input_folder, output_folder, augmentation, args):
    """
    Augments data from input_folder using the provided augmentation function and saves to output_folder.

    Args:
        input_folder (str): Path to the folder containing input files.
        output_folder (str): Path to the folder where augmented files will be saved.
        augmentation (callable): Function that takes (input_file, output_file, args) and performs augmentation.
        args: Additional arguments to pass to the augmentation function.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over files in input_folder
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if os.path.isfile(input_file):
            # Create output file path with same filename
            output_file = os.path.join(output_folder, filename)
            # Apply augmentation
            augmentation(input_file, output_file, args)


def randomly_eq(input_file, output_file, args):
    """
    Applies random equalization to an audio file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        args (dict): Dictionary containing parameters:
            - 'sr': Sampling rate (default: 22050)
            - 'bands': List of frequency band tuples [(low, high), ...] (default: [(20, 150), (150, 800), (800, 3000), (3000, 8000)])
            - 'gain_range': Tuple (min_gain, max_gain) in dB (default: (-6, 6))
    """
    # Default parameters
    sr = args.get('sr', 22050)
    default_bands = [(100, 500), (500, 2000), (2000, 8000)]  # Safer default bands
    bands = args.get('bands', default_bands)
    gain_range = args.get('gain_range', (-6, 6))
    
    # Load audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # Adjust bands to be within valid frequency range for the loaded sampling rate
    nyquist = sr_loaded / 2
    adjusted_bands = []
    for low, high in bands:
        # Ensure frequencies don't exceed Nyquist frequency and are positive
        adj_low = max(1.0, min(float(low), nyquist - 1))  # At least 1 Hz, leave small margin
        adj_high = min(float(high), nyquist - 1)  # Leave small margin
        if adj_low < adj_high and adj_high > 0:
            adjusted_bands.append((adj_low, adj_high))
    
    # If no valid bands after adjustment, use default safe bands
    if not adjusted_bands:
        adjusted_bands = [(100.0, nyquist * 0.1), (nyquist * 0.1, nyquist * 0.4), (nyquist * 0.4, nyquist * 0.8)]
    
    bands = adjusted_bands
    
    # Apply random EQ by filtering each band and applying random gain
    y_eq = np.zeros_like(y)
    
    for low, high in bands:
        # Design bandpass filter
        sos = butter(2, [low, high], btype='bandpass', fs=sr_loaded, output='sos')
        
        # Filter the signal
        y_band = sosfilt(sos, y)
        
        # Apply random gain to this band
        gain_db = np.random.uniform(gain_range[0], gain_range[1])
        gain_linear = 10 ** (gain_db / 20)
        y_band *= gain_linear
        
        # Add to output
        y_eq += y_band
    
    # Normalize to prevent clipping
    y_eq = librosa.util.normalize(y_eq)
    
    # Save augmented audio
    _write_mp3(output_file, y_eq, sr_loaded)


def apply_reverb(input_file, output_file, args):
    """
    Applies convolutional reverb to an audio file using an impulse response.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        args (dict): Dictionary containing parameters:
            - 'ir_path': Path to the impulse response audio file
            - 'sr': Sampling rate (default: 22050)
            - 'wet_dry_mix': Ratio of wet to dry signal (0.0 = dry only, 1.0 = wet only, default: 0.3)
    """
    # Default parameters
    ir_path = args.get('ir_path')
    if ir_path is None:
        raise ValueError("ir_path must be provided in args")
    sr = args.get('sr', 22050)
    wet_dry_mix = args.get('wet_dry_mix', 0.3)
    
    # Load input audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # Load impulse response TODO: load at any sr and convert to match input audio sr
    ir, ir_sr = librosa.load(ir_path, sr=sr_loaded)
    
    # Apply convolution for reverb
    y_wet = convolve(y, ir, mode='full')
    
    # Trim to original length
    y_wet = y_wet[:len(y)]
    
    # Mix dry and wet signals
    y_reverb = (1 - wet_dry_mix) * y + wet_dry_mix * y_wet
    
    # Normalize to prevent clipping
    y_reverb = librosa.util.normalize(y_reverb)
    
    # Save augmented audio
    _write_mp3(output_file, y_reverb, sr_loaded)


def apply_compression(input_file, output_file, args):
    """
    Applies dynamic compression to an audio file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        args (dict): Dictionary containing parameters:
            - 'sr': Sampling rate (default: 22050)
            - 'threshold_db': Compression threshold in dB (default: -20)
            - 'ratio': Compression ratio (default: 4.0, higher = more compression)
            - 'attack_ms': Attack time in milliseconds (default: 5)
            - 'release_ms': Release time in milliseconds (default: 100)
            - 'makeup_gain_db': Makeup gain in dB (default: 0)
    """
    # Default parameters
    sr = args.get('sr', 22050)
    threshold_db = args.get('threshold_db', -20)
    ratio = args.get('ratio', 4.0)
    attack_ms = args.get('attack_ms', 5)
    release_ms = args.get('release_ms', 100)
    makeup_gain_db = args.get('makeup_gain_db', 0)
    
    # Load audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # Convert parameters
    threshold_linear = 10 ** (threshold_db / 20)
    attack_samples = int(attack_ms * sr_loaded / 1000)
    release_samples = int(release_ms * sr_loaded / 1000)
    makeup_gain = 10 ** (makeup_gain_db / 20)
    
    # Simple compressor implementation
    y_compressed = np.zeros_like(y)
    envelope = np.zeros_like(y)
    
    for i in range(len(y)):
        # Calculate envelope (smoothed absolute value)
        abs_sample = abs(y[i])
        if abs_sample > envelope[max(0, i-1)]:
            # Attack
            envelope[i] = envelope[max(0, i-1)] + (abs_sample - envelope[max(0, i-1)]) / attack_samples
        else:
            # Release
            envelope[i] = envelope[max(0, i-1)] * (1 - 1/release_samples)
        
        # Calculate gain reduction
        if envelope[i] > threshold_linear:
            gain_reduction = (envelope[i] / threshold_linear) ** (1/ratio - 1)
            gain = 1 / gain_reduction
        else:
            gain = 1.0
        
        # Apply compression
        y_compressed[i] = y[i] * gain * makeup_gain
    
    # Normalize to prevent clipping
    y_compressed = librosa.util.normalize(y_compressed)
    
    # Save augmented audio
    _write_mp3(output_file, y_compressed, sr_loaded)


def add_noise(input_file, output_file, args):
    """
    Adds additive noise to an audio file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        args (dict): Dictionary containing parameters:
            - 'sr': Sampling rate (default: 22050)
            - 'noise_type': Type of noise ('white', 'pink', 'brown', default: 'white')
            - 'snr_db': Target signal-to-noise ratio in dB (default: 25, higher = cleaner)
    """
    # Default parameters
    sr = args.get('sr', 22050)
    noise_type = args.get('noise_type', 'white')
    snr_db = float(args.get('snr_db', 25))
    
    # Load audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # SNR must be positive for the "signal/noise" convention used below.
    # Accept negative user input for backwards compatibility, but map it to magnitude.
    target_snr_db = abs(snr_db)

    # Calculate noise level based on target SNR.
    signal_power = np.mean(y ** 2)
    if signal_power <= 1e-12:
        # Silence or near-silence: nothing meaningful to augment.
        _write_mp3(output_file, y, sr_loaded)
        return
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    
    # Generate noise
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    elif noise_type == 'pink':
        # Simple pink noise approximation
        white = np.random.normal(0, 1, len(y))
        noise = np.convolve(white, [1, 1], mode='same')
        noise = noise / (np.std(noise) + 1e-12) * np.sqrt(noise_power)
    elif noise_type == 'brown':
        # Brown noise (integrated white noise)
        white = np.random.normal(0, 1, len(y))
        noise = np.cumsum(white)
        noise = noise / (np.std(noise) + 1e-12) * np.sqrt(noise_power)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Add noise to signal
    y_noisy = y + noise
    
    # Preserve loudness/SNR by only scaling when clipping would occur.
    peak = np.max(np.abs(y_noisy))
    if peak > 1.0:
        y_noisy = y_noisy / peak
    
    # Save augmented audio
    _write_mp3(output_file, y_noisy, sr_loaded)
