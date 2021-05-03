import numpy as np
from scipy.io.wavfile import read as audioread
import matplotlib
from matplotlib import pyplot as plt


def make_sine_wavetable(n_samples, phases, amps, freqs):
    """Makes a wavetable from a sum of sines."""
    t = np.linspace(0, 1, num=n_samples)
    wavetable = np.zeros_like(t)
    for amp, phase, freq in zip(amps,
                                phases,
                                freqs):
        wavetable += amp * np.sin(np.sin(2 * np.pi * freq * t + phase)) + \
                         amp / 2 * np.sin(np.sin(2 * np.pi * 2 * freq * t + phase))
    return wavetable

def synthesize(sampling_speed, wavetable, n_samples):
      """Synthesizes a new waveform from an existing wavetable."""
      samples = []
      current_sample = 0
      while len(samples) < n_samples:
          current_sample += sampling_speed
          current_sample = current_sample % wavetable.size
          samples.append(wavetable[current_sample])
          current_sample += 1
      return np.array(samples)


# reference https://flothesof.github.io/Karplus-Strong-algorithm-Python.html
