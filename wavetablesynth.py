import numpy as np


t = np.linspace(0, 1, num=100)
wavetable = np.sin(np.sin(2 * np.pi * t)) # sine wave
wavetable1 = t * (t < 0.5) + (-(t - 1)) * (t>= 0.5) #triangle wave


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

def synthesize1(sampling_speed, wavetable1, n_samples):
    """Synthesizes a new waveform from an existing wavetable."""
    samples = []
    current_sample = 0
    while len(samples) < n_samples:
        current_sample += sampling_speed
        current_sample = current_sample % wavetable.size
        samples.append(wavetable[current_sample])
        current_sample += 1
    return np.array(samples)


sample1 = synthesize(440, wavetable, 2*fs)
sample2 = synthesize1(440, wavetable1, 2 *fs)
