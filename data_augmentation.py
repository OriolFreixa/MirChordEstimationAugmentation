import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, convolve

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
            - 'bands': List of frequency band tuples [(low, high), ...] (default: [(20, 250), (250, 2000), (2000, 8000), (8000, 20000)])
            - 'gain_range': Tuple (min_gain, max_gain) in dB (default: (-6, 6))
    """
    # Default parameters
    sr = args.get('sr', 22050)
    default_bands = [(20, 150), (150, 800), (800, 3000), (3000, 8000)]  # Much safer default bands
    bands = args.get('bands', default_bands)
    gain_range = args.get('gain_range', (-6, 6))
    
    # Load audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # Adjust bands to be within valid frequency range for the loaded sampling rate
    nyquist = sr_loaded / 2
    adjusted_bands = []
    for low, high in bands:
        # Ensure frequencies don't exceed Nyquist frequency and are positive
        adj_low = max(1.0, min(float(low), nyquist - 1000))  # At least 1 Hz, leave large margin
        adj_high = min(float(high), nyquist - 500)  # Leave large margin
        if adj_low < adj_high and adj_high > 0:
            adjusted_bands.append((adj_low, adj_high))
    
    # If no valid bands after adjustment, use default safe bands
    if not adjusted_bands:
        adjusted_bands = [(20.0, nyquist * 0.05), (nyquist * 0.05, nyquist * 0.2), (nyquist * 0.2, nyquist * 0.6)]
    
    bands = adjusted_bands
    
    # Apply random EQ to each band
    y_eq = np.zeros_like(y)
    for low, high in bands:
        try:
            # Normalize frequencies to 0-1 range
            low_norm = low / (sr_loaded / 2)
            high_norm = high / (sr_loaded / 2)
            
            # Ensure normalized frequencies are in valid range
            if low_norm >= 1.0 or high_norm >= 1.0 or low_norm <= 0 or high_norm <= 0:
                print(f"Warning: Invalid normalized frequencies {low_norm}, {high_norm} for band ({low}, {high})")
                continue
                
            # Design bandpass filter with normalized frequencies
            sos = butter(8, [low_norm, high_norm], 'bandpass', output='sos')
            # Filter the signal
            y_band = sosfilt(sos, y)
            # Apply random gain
            gain_db = np.random.uniform(gain_range[0], gain_range[1])
            gain_linear = 10 ** (gain_db / 20)
            y_band *= gain_linear
            # Add to output
            y_eq += y_band
        except Exception as e:
            print(f"Warning: Could not process band ({low}, {high}): {e}")
            # Skip this band
            continue
    
    # Normalize to prevent clipping
    y_eq = librosa.util.normalize(y_eq)
    
    # Save augmented audio
    sf.write(output_file, y_eq, sr_loaded)


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
    sf.write(output_file, y_reverb, sr_loaded)


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
    sf.write(output_file, y_compressed, sr_loaded)


def add_noise(input_file, output_file, args):
    """
    Adds additive noise to an audio file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the augmented audio file.
        args (dict): Dictionary containing parameters:
            - 'sr': Sampling rate (default: 22050)
            - 'noise_type': Type of noise ('white', 'pink', 'brown', default: 'white')
            - 'snr_db': Signal-to-noise ratio in dB (default: -20, lower = more noise)
    """
    # Default parameters
    sr = args.get('sr', 22050)
    noise_type = args.get('noise_type', 'white')
    snr_db = args.get('snr_db', -1.0)
    
    # Load audio
    y, sr_loaded = librosa.load(input_file, sr=sr)
    
    # Calculate noise level based on SNR
    signal_power = np.mean(y ** 2)
    noise_power = signal_power // (10 ** (snr_db // 10))
    
    # Generate noise
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    elif noise_type == 'pink':
        # Simple pink noise approximation
        white = np.random.normal(0, 1, len(y))
        noise = np.convolve(white, [1, 1], mode='same')
        noise = noise / np.std(noise) * np.sqrt(noise_power)
    elif noise_type == 'brown':
        # Brown noise (integrated white noise)
        white = np.random.normal(0, 1, len(y))
        noise = np.cumsum(white)
        noise = noise / np.std(noise) * np.sqrt(noise_power)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Add noise to signal
    y_noisy = y + noise
    
    # Normalize to prevent clipping
    y_noisy = librosa.util.normalize(y_noisy)
    
    # Save augmented audio
    sf.write(output_file, y_noisy, sr_loaded)