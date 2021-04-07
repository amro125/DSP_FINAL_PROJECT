#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#from IPython.display import Audio
from scipy.signal import sawtooth, square, butter, filtfilt
from scipy.io.wavfile import read


# In[2]:


def lowpass(x, cutoff, order, fs):
    nyquistRate = 0.5 * fs
    normal_cutoff = cutoff / nyquistRate
    (b,a) = butter(order, normal_cutoff, btype = 'low', analog = False)
    filtered_sig = filtfilt(b,a,x)

    return filtered_sig


# In[3]:


def butter_highpass(cutoff, fs, order):
    nyquistRate = 0.5 * fs
    normal_cutoff = cutoff / nyquistRate
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


# In[4]:


def butter_highpass_filter(x, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order)
    filtered_sig1 = signal.filtfilt(b, a, x)
    return filtered_sig1


# In[ ]:
