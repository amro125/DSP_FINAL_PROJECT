# from https://publish.illinois.edu/augmentedlistening/tutorials/music-processing/tutorial-3-amplitude-clipping-effects/

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal




def distortion(audioin, amplification, cAudioFilePath):
    fs, audioin = read(cAudioFilePath)
    audioin = audioin[:,0]
    th = max(audioin)
    out = audioin * amplification
    for i in range(len(out)):
        if(abs(out[i])>th):
            out[i] = abs(out[i])/out[i]*th
    return out

#hardclip_out = distortion(audioin, 20.5, "ProjectStudiomvmtI final.wav")

def overdrive(audioin):

    out = np.arctan(audioin)
    return out

#softclip_out = overdrive(audioin)
