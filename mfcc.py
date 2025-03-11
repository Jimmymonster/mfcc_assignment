import numpy as np
import scipy.signal
import scipy.fftpack
import librosa

def pre_emphasis(signal, coeff=0.97):
    """Applies pre-emphasis to the signal."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """Splits the signal into overlapping frames."""
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    num_frames = int(np.ceil(float(len(signal) - frame_length + frame_step) / frame_step))

    pad_signal_length = frame_length + (num_frames - 1) * frame_step
    pad_signal = np.pad(signal, (0, max(0, pad_signal_length - len(signal))), mode='constant')

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32)]
    return frames

def windowing(frames):
    """Applies a Hann window to each frame."""
    return frames * np.hanning(frames.shape[1])

def fft(frames, NFFT=512):
    """Computes the power spectrum of each frame."""
    power_spectrum = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    return power_spectrum

def mel_filterbank(fft_frames, sample_rate, nfilt=40, NFFT=512):
    """Applies a mel filterbank to the power spectrum."""
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    for i in range(1, nfilt + 1):
        left, center, right = bin[i - 1], bin[i], bin[i + 1]
        if left < center:
            fbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if center < right:
            fbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    filterbank_energies = np.dot(fft_frames, fbank.T)
    filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)
    return librosa.power_to_db(filterbank_energies)  # Log compression using librosa

def dct(log_energies, num_mfcc=12, lifter=22):
    """Computes the Discrete Cosine Transform (DCT) and applies liftering."""
    mfccs = scipy.fftpack.dct(log_energies, type=2, axis=1, norm='ortho')[:, 1:num_mfcc + 1]
    
    # Apply liftering (cepstral smoothing)
    (nframes, ncoeff) = mfccs.shape
    n = np.arange(ncoeff)
    lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
    return mfccs * lift

def mfcc(signal, sample_rate, num_mfcc=12):
    """Computes MFCC features from an audio signal."""
    # Step 1: Pre-emphasis (optional, librosa does not use it by default)
    # emphasized_signal = pre_emphasis(signal)  # Uncomment if needed

    # Step 2: Framing
    frames = framing(signal, sample_rate)

    # Step 3: Windowing (use Hann window like librosa)
    windowed_frames = windowing(frames)

    # Step 4: FFT and Power Spectrum
    fft_frames = fft(windowed_frames)

    # Step 5: Mel Filterbank
    filterbank_energies = mel_filterbank(fft_frames, sample_rate)

    # Step 6: DCT with liftering
    mfcc_features = dct(filterbank_energies, num_mfcc)

    return mfcc_features
