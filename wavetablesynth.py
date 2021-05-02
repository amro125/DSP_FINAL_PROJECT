import numpy as np
from scipy.io.wavfile import read as audioread

#t = np.linspace(0, 1, num=100)
#wavetable = np.sin(np.sin(2 * np.pi * t)) # sine wave
#wavetable1 = t * (t < 0.5) + (-(t - 1)) * (t>= 0.5) #triangle wave


def synthesize(sampling_speed, n_samples, cAudioFilePath):
    """Synthesizes a new waveform from an existing wavetable."""
    fs, x = audioread(cAudioFilePath)
    t = np.linspace(0, 1, num=100)
    wavetable = np.sin(np.sin(2 * np.pi * t)) # sine wave
    samples = []
    current_sample = 0
    while len(samples) < n_samples:
        current_sample += sampling_speed
        current_sample = current_sample % wavetable.size
        samples.append(wavetable[current_sample])
        current_sample += 1
    return np.array(samples)

def synthesize1(sampling_speed, n_samples, cAudioFilePath):
    """Synthesizes a new waveform from an existing wavetable."""
    fs, x = audioread(cAudioFilePath)
    t = np.linspace(0, 1, num=100)
    wavetable1 = t * (t < 0.5) + (-(t - 1)) * (t>= 0.5) #triangle wave
    samples = []
    current_sample = 0
    while len(samples) < n_samples:
        current_sample += sampling_speed
        current_sample = current_sample % wavetable1.size
        samples.append(wavetable1[current_sample])
        current_sample += 1
    return np.array(samples)

# Reference: https://flothesof.github.io/Karplus-Strong-algorithm-Python.html

