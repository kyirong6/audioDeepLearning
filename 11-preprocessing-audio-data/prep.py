import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # signalArray = sr * T = 22050 * 30s (30s length of song)
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)
# magnitude indicates the contribution of the frequency to
# the overall sound
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

# first half of frequency/magnitude array
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# number of samples per fft (window)
n_fft = 2048

# amount we are shifting each fourier transform to the right
# 512 samples to the right
hop_length = 512

# short term ft
# stft -> spectrogram
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

# convert amplitude to decibels via log transformation
# makes the heat map much more interpretable
log_spectrogram = librosa.amplitude_to_db(spectrogram)

# similar to a heat map
# color represents amplitude in dicibels
# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
