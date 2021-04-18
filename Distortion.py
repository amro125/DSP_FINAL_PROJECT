import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal


fs, x = read()
x = x[:,0]


def distortion(audioin, amplification):

    th = max(audioin)
    out = audioin * amplification
    for i in range(len(out)):
        if(abs(out[i])>th):
            out[i] = abs(out[i])/out[i]*th
    return out

hardclip_out = distortion(x, 20.5)

def overdrive(audioin):

    out = np.arctan(audioin)
    return out

softclip_out = overdrive(x)
