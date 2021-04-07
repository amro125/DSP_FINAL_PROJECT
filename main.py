# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# The Fundamental Frequencies: A=55, r=1.0594 = exp(log(2)/12)


import math
import wave
import struct
import sys
import numpy as np

C = 32.7
Csharp = Dflat = 34.65
D = 36.71
Dsharp = Eflat = 38.89
E = 41.20
F = 43.65
Fsharp = Gflat = 46.25
G = 49
Gsharp = Aflat = 51.91
A = 55
Asharp = Bflat = 58.27
B = 61.74
C2 = 65.41

frate = 11025
filename = 'test.wav'
def save_as_wav(file, lst, size):
  amp = 8000.0
  nchannels = 1
  samples = 2
  wav_file = wave.open(file, "w")
  wav_file.setparams((nchannels, samples, frate, size, "NONE", "not compressed"))
  print("Generating file")
  for s in lst:
      wav_file.writeframes(struct.pack('h', int(s*amp/2)))
  wav_file.close()
  print("Saved to ", file)


def generate_wave_input(freq, length, rate=44100, phase=0.0):

    length = int(length * rate)
    t = np.arange(length) / float(rate)
    omega = float(freq) * 2 * math.pi
    phase *= 2 * math.pi
    return omega * t + phase

def sine(freq, length, rate=3200, phase=0.0):

    data = generate_wave_input(freq, length, rate, phase)
    return np.sin(data)

def makesound(freq,octave, amp, size):
    octavef = [x * octave for x in freq]
    sines = [np.sin( math.pi * y * x / frate) for y in octavef for x in range(size)]
    return sines

def feedback_modulated_delay(data, modwave, dry, wet):

    out = data.copy()
    for i in range(len(data)):
        index = int(i - modwave[i])
        if index >= 0 and index < len(data):
            out[i] = out[i] * dry + out[index] * wet
    return out

def flanger(data, freq, dry=0.5, wet=0.5, depth=20.0, delay=1.0, rate=3200):
    length = float(len(data)) / rate
    mil = float(rate) / 1000
    delay *= mil
    depth *= mil
    modwave = (sine(freq, length) / 2 + 0.5) * depth + delay
    print(modwave)
    return feedback_modulated_delay(data, modwave, dry, wet)

note_list = [E,D,C,D,E,F,E,F,G,F,E,C,C,C]

signal = makesound(note_list, 12, 4, 3200)
Flanged = flanger(signal,440.0)

save_as_wav(filename, Flanged, 3200)