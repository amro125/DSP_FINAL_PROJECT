

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal




def distortion(data, amp, cAudioFilePath):
    fs, data = read(cAudioFilePath)
    data = data[:,0]
    th = max(data)
    out = data * amp
    for i in range(len(out)):
        if(abs(out[i])>th):
            out[i] = abs(out[i])/out[i]*th
    return out

#hardclip_out = distortion(audioin, 20.5, "ProjectStudiomvmtI final.wav")

def overdrive(data):

    out = np.arctan(data)
    return out

#softclip_out = overdrive(audioin)
#Reference: https://publish.illinois.edu/augmentedlistening/tutorials/music-processing/tutorial-3-amplitude-clipping-effects/
