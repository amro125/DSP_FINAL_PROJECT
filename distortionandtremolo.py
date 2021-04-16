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


    def distortion(x, fs):
        th = max(fs)
        out = x * fs
        for i in range(len(out)):
            if(abs(out[i])>th):
                out[i] = abs(out[i])/out[i]*th
        print(out)

    distortion(x, fs)
