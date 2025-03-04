import numpy as np
import scipy.signal
import scipy.fftpack
import librosa
import librosa.display
import matplotlib.pyplot as plt

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    num_frames = int(np.ceil(float(len(signal) - frame_length + frame_step) / frame_step))

    # Ensure padding covers all frames
    pad_signal_length = frame_length + (num_frames - 1) * frame_step
    pad_signal = np.append(signal, np.zeros((pad_signal_length - len(signal))))

    # Compute frame indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    # Extract frames
    frames = pad_signal[indices.astype(np.int32)]
    
    return frames

def windowing(frames):
    hamming = np.hamming(frames.shape[1])
    return frames * hamming

def fft(frames, NFFT=512):
    return np.abs(np.fft.rfft(frames, NFFT)) / NFFT  # Normalize FFT

def mel_filterbank(fft_frames, sample_rate, nfilt=40, NFFT=512):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    for i in range(1, nfilt + 1):
        left, center, right = bin[i - 1], bin[i], bin[i + 1]
        for j in range(left, center):
            fbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i - 1, j] = (right - j) / (right - center)
    
    filterbank_energies = np.dot(fft_frames, fbank.T)
    filterbank_energies = np.log(filterbank_energies + np.finfo(float).eps)  # Apply log here
    
    return filterbank_energies

def dct(log_energies, num_ceps=12):
    return scipy.fftpack.dct(log_energies, type=2, axis=1, norm='ortho')[:, 1:num_ceps+1]  # Remove first coefficient

def mfcc(signal, sample_rate):
    emphasized_signal = pre_emphasis(signal)
    frames = framing(emphasized_signal, sample_rate)
    windowed_frames = windowing(frames)
    fft_frames = fft(windowed_frames)
    filterbank_energies = mel_filterbank(fft_frames, sample_rate)
    mfcc_features = dct(filterbank_energies)
    return mfcc_features

# Example usage
if __name__ == "__main__":
    signal, sr = librosa.load("test.wav", sr=16000)
    mfcc_features = mfcc(signal, sr)
    print(mfcc_features.shape)
    
    # Plot MFCC
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfcc_features, x_axis="time", cmap="viridis", sr=sr)
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC Features")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()
