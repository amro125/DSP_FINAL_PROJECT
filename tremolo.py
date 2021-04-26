import scipy
from scipy.io import wavfile
import math
from math import exp



def tremolo( amp, speed, x, fs=44100):
    time = len(x)/fs
    k = np.arange(0,time,1/fs)
    lfo = amp*np.sin(2*np.pi*speed*k) #creates lfo
    tremoloI = (x*lfo)                #applies lfo
    return tremoloI

