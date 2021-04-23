import numpy as np
from scipy import interpolate

new_rate = 22050
fs, x = wavfile.read("")

if fs != new_rate:
    duration = x.shape[0] / fs

    time_old = np.linspace(0, duration, x.shape[0])
    time_new = np.linspace(0, duration, int(x.shape[0] * new_rate/fs))
    interpolator = interpolate.interp1d(time_old, x.T)
    new_audio = interpolator(time_new).T

    wavfile.write("out1.wav", new_rate, np.round(new_audio).astype(x.dtype))
