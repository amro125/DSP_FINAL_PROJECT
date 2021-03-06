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


def lowpassfilt(x, cutoff, order, fs):
    nyqRate = 0.5 * fs
    original_cutoff = cutoff / nyqRate
    (b,a) = butter(order, original_cutoff, btype = 'low', analog = False)
    filtered_sig = filtfilt(b,a,x)

    return filtered_sig

#Reference: https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7

# In[3]:


def butter_highpass(cutoff, fs, order):
    nyqRate = 0.5 * fs
    original_cutoff = cutoff / nyqRate
    (b, a) = butter(order, original_cutoff, btype='high', analog=False)
    return b, a


# In[4]:


def butter_highpass_filter(x, cutoff, fs, order):
    (b, a) = butter_highpass(cutoff, fs, order)
    filtered_sig1 = signal.filtfilt(b, a, x)
    return filtered_sig1


# In[ ]:
