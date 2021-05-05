import numpy as np
from scipy.io.wavfile import write
from scipy import signal as sg

#data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
def bitdepth16(sampling_rate, freq, duration):
    #sampling_rate = 44100                    ## Sampling Rate
    #freq = 150                               ## Frequency (in Hz)
    #duration = 3   # in seconds, may be float
    t = np.linspace(0, duration, sampling_rate*duration) # Creating time vector
    data = sg.sawtooth(2 * np.pi * freq * t, 0)          # Sawtooth signal
    scaled = np.int16(data/np.max(np.abs(data)) * 32767) 
    write('test1.wav', 44100, scaled) # Write to file. Can be overridden
    
bitdepth16(44100, 150, 3)



#data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
def bitdepth8(sampling_rate, freq, duration):
    #sampling_rate = 44100                    ## Sampling Rate
    #freq = 150                               ## Frequency (in Hz)
    #duration = 3   # in seconds, may be float
    t = np.linspace(0, duration, sampling_rate*duration) # Creating time vector
    data = sg.sawtooth(2 * np.pi * freq * t, 0)          # Sawtooth signal
    scaled = np.int8(data/np.max(np.abs(data)) * 256) 
    write('test2.wav', 44100, scaled) # Write to file. Can be overridden
    
bitdepth8(44100, 150, 3)
