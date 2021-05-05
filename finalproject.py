# DSP Final Project.
# Amit Rogel and Lauren McCall


import math
import wave
import struct
import sys
import numpy as np
import scipy
from scipy.io.wavfile import read
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import fftpack
from scipy import interpolate, signal
from scipy.signal import sawtooth, square, butter, filtfilt
from scipy.io.wavfile import write
from scipy.io import wavfile


# fundamental frequencies
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
note_list = []

# float(sampleRate.get())

def save_as_wav(file, lst, size, amp):
    nchannels = 1
    samples = 2
    wav_file = wave.open(file, "w")
    frate = int(sampleRate.get())
    wav_file.setparams((nchannels, samples, frate, size, "NONE", "not compressed"))
    print("Generating file")
    for s in lst:
        wav_file.writeframes(struct.pack('h', int(s * amp / 2)))
    wav_file.close()
    print("Saved to ", file)


def sine(freq, length, phase=0.0):  # Makes a sine wave to use for effects
    rate = int(sampleRate.get())
    length = int(length * rate)
    t = np.arange(length) / float(rate)
    omega = float(freq) * 2 * math.pi
    phase *= 2 * math.pi
    data = omega * t + phase
    return np.sin(data)


def signalMaker(freq, octave, amp, size):  # Checks the typ of signal to generate
    setting = wavestyle.get()
    for i in effects:
        i.configure(bg='grey')
    sample = int(sampleRate.get())
    octave = 16
    global signal
    if setting == "Additive Sin":
        signal = makesoundsin(note_list, octave, 4, sample)
    if setting == "Additive Sawtooth":
        signal = makesoundsaw(note_list, octave, 4, sample)
    if setting == "Additive Square":
        signal = makesoundsquare(note_list, octave, 4, sample)
    if setting == "Wavetable":
        waveT = ywave
        signal = wavetable(note_list, waveT, sample, octave)
    return


def makesoundsaw(freq, octave, amp, size):
    # Additive Synth sawtooth wave
    octavef = [x * octave for x in freq]
    sines = [scipy.signal.sawtooth(math.pi * y * x / int(sampleRate.get())) for y in octavef for x in range(size)]
    return sines

def makesoundsquare(freq, octave, amp, size):
    # Additive Synth square wave
    octavef = [x * octave for x in freq]
    sines = [scipy.signal.square(math.pi * y * x / int(sampleRate.get())) for y in octavef for x in range(size)]
    return sines


def makesoundsin(freq, octave, amp, size):
    # Additive Synth sound part
    octavef = [x * octave for x in freq]
    sines = [np.sin(math.pi * y * x / int(sampleRate.get())) for y in octavef for x in range(size)]
    return sines


def wavetable(freq, wavetable, n_samples, octave):
    # Takes drawn wave and makes into sound based on note list
    # Algorithm referenced from https://flothesof.github.io/Karplus-Strong-algorithm-Python.html
    samples = []
    sig =[]
    current_sample = 0
    new_x = np.concatenate([np.linspace(xwave[i], xwave[i + 1], num=30) for i in range(len(xwave) - 1)])
    wavetable = np.interp(new_x, xwave, ywave)
    if len(wavetable) != float(sampleRate.get()):
        wavetablebig = resampling(wavetable, int(sampleRate.get()), len(wavetable))
        print( "resaamples", len(wavetable), "to", int(sampleRate.get()))
    print(len(wavetablebig))
    octavef = [x * octave for x in freq]
    for sampling_speed in octavef:
        samples.clear()
        while len(samples) < n_samples:
            current_sample += int(sampling_speed)
            current_sample = current_sample % wavetablebig.size
            samples.append(wavetablebig[current_sample])
            current_sample += 1
        sig += samples
    return sig


def feedback_modulated_delay(data, modwave, dry, wet):
    # Feedback for flanger
    out = data.copy()
    for i in range(len(data)):
        index = int(i - modwave[i])
        if index >= 0 and index < len(data):
            out[i] = out[i] * dry + out[index] * wet
    return out


def flanger(data, freq, dry=0.5, wet=0.5, depth=20.0, delay=1.0, rate=3200):
    # Referenced from Wybiral Github https://github.com/wybiral/python-musical/blob/master/musical/audio/effect.py
    flangerbutton.configure(bg="green")
    length = float(len(data)) / rate
    mil = float(rate) / 1000
    delay *= mil
    depth *= mil
    modwave = (sine(freq, length) / 2 + 0.5) * depth + delay
    global signal
    print(modwave)
    signal = feedback_modulated_delay(data, modwave, dry, wet)
    return


def lowpass(x, cutoff, order):
    # referenced from https://flothesof.github.io/Karplus-Strong-algorithm-Python.html
    lpf.configure(bg="green")
    nyquistRate = 0.5 * int(sampleRate.get())
    normal_cutoff = cutoff / nyquistRate
    (b, a) = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_sig = filtfilt(b, a, x)
    global signal
    signal = filtfilt(b, a, x)
    return


def butter_highpass(cutoff, fs, order):
    nyquistRate = 0.5 * fs
    normal_cutoff = cutoff / nyquistRate
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(x, cutoff, order):
    hpf.configure(bg="green")
    fs = int(sampleRate.get())
    b, a = butter_highpass(cutoff, fs, order)
    global signal
    signal = scipy.signal.filtfilt(b, a, x)
    return


def tremolo(x, amp, speed):  #Tremolo effect function creation
    tremolobutton.configure(bg="green")
    fs = int(sampleRate.get())
    time = len(x)/fs
    k = np.arange(0, time, 1/fs)
    lfo = amp*np.sin(2*np.pi*speed*k)  # creates lfo
    if len(lfo) != len(x):
       x=np.pad(x, (0, (len(lfo)-len(x))), 'edge')
    print(len(lfo),len(x))
    tremoloI = (x*lfo)
    global signal
    signal = tremoloI
    return


def distortion(audioin, amplification):  # manipulates based on arctan
    distortionbutton.configure(bg='green')
    th = max(audioin)
    out = []
    out.clear()
    for i in audioin:
        out.append(np.arctan(i*amplification)/np.arctan(amplification))
    global signal
    length = float(len(out))/float(sampleRate.get())
    signal = out*sine(262, length)
    return


def reverb(audio, echodur, delay_amp, sample):  # Convolution Reverb
    reverbbutton.configure(bg='green')
    sample = int(sampleRate.get())
    delay_len_samples = round(echodur * sample)
    leading_zero_padding_sig = np.zeros(delay_len_samples)
    impulse_response = np.zeros(delay_len_samples)
    impulse_response[0] = 1
    impulse_response[-1] = delay_amp
    global signal
    signal = np.convolve(audio, impulse_response)


def resampling(audio, new_rate, fs):  # Resampling
    x = audio
    lenx = len(x)
    if fs != new_rate:
                duration = lenx / fs
                time_old = np.linspace(0, duration, lenx)
                time_new = np.linspace(0, duration, int(lenx * new_rate/fs))
                interpolator = interpolate.interp1d(time_old, x)
                new_audio = interpolator(time_new).T
    sampleRate.delete(0,END)
    sampleRate.insert(0,new_rate)
    global signal
    signal = new_audio
    return new_audio


def bitchange(x):
    bit = bitrate.get()
    global signal
    if bit == "8 bit":
        scaled = np.int8(x/np.max(np.abs(x)) * 255)  # downquantizes based on 2^8 -1
        file = 'SunshineSynth8bit.wav'
    if bit == "16 bit":
        scaled = np.int8(x /np.max(np.abs(x)) * 32767)  # downquantizes based on 2^15 -1
        print(scaled)
        file = 'SunshineSynth16bit.wav'
    save_as_wav(file, scaled, int(sampleRate.get()), amp=100.0)
    # write(file, int(sampleRate.get()), scaled)  # Write to file.
    signal = scaled
    print("saved new bit to ", file)

    return


def noteClick(number, letter):  # adds notes to list
    # e.delete(0, END)
    note_list.append(number)
    current = e.get()
    e.delete(0, END)
    e.insert(0, str(current) + letter)


def clearwave():  # this clears the wavetable drawing
    canvas.delete("all")
    xpos.clear()
    ypos.clear()


def clear():  # This clears all the note inputs for wavetable and additive
    e.delete(0, END)
    note_list.clear()
    canvas.delete("all")
    xpos.clear()
    ypos.clear()
    for i in effects:
        i.configure(bg='grey')
    print("Reset!")


def makegraph(audio):  # Makes a graph of the signal
    fig1 = plt.figure(figsize=(20, 20))
    plt.plot(audio)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Sound Generated')
    plt.show()


def makefft(sig): # Makes an fft graph
    xfft = np.fft.fft(sig)  # takes fft
    xabs = np.abs(xfft)  # takes magnintude
    xphase = np.angle(xfft)  # takes phase

    # Plot the FFT
    plt.figure(figsize=(6, 5))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(xabs)
    ax1.set_title('Magnitude and Phase Graph')
    ax1.set(ylabel='Magnitude')
    ax2.plot(xphase)
    ax2.set(xlabel='Frequency (Hz)', ylabel='Phase')
    fig.savefig('Q3_square.jpg')
    plt.show()


root = Tk()
root.title("Sunshine Synth")


# Set up Note Buttons
ButtonC = Button(root, text="C", padx=30, pady=20, command=lambda: noteClick(C, "C "))
ButtonCsharp = Button(root, fg="white", bg="black", text="C#/Db", padx=20, pady=20, command=lambda: noteClick(Csharp, "C# "))
ButtonD = Button(root, text="D", padx=30, pady=20, command=lambda: noteClick(D, "D "))
ButtonDsharp = Button(root, fg="white", bg="black", text="D#/Eb", padx=20, pady=20, command=lambda: noteClick(C, "D# "))
ButtonE = Button(root, text="E", padx=30, pady=20, command=lambda: noteClick(E, "E "))
ButtonF = Button(root, text="F", padx=30, pady=20, command=lambda: noteClick(F, "F "))
ButtonFsharp = Button(root, fg="white", bg="black", text="F#/Gb", padx=20, pady=20, command=lambda: noteClick(Fsharp, "F# "))
ButtonG = Button(root, text="G", padx=30, pady=20, command=lambda: noteClick(G, "G "))
ButtonGsharp = Button(root, fg="white", bg="black", text="G#/Ab", padx=20, pady=20, command=lambda: noteClick(Gsharp, "G# "))
ButtonA = Button(root, text="A", padx=30, pady=20, command=lambda: noteClick(A, "A "))
ButtonAsharp = Button(root, fg="white", bg="black", text="A#/Bb", padx=20, pady=20, command=lambda: noteClick(Asharp, "A# "))
ButtonB = Button(root, text="B", padx=30, pady=20, command=lambda: noteClick(B, "B "))
ButtonC2 = Button(root, text="C2", padx=30, pady=20, command=lambda: noteClick(C2, "C2 "))
ButtonClear = Button(root, text="Clear", padx=30, pady=20, command=clear)


# Put buttons into grid
ButtonC.grid(row=4, column=0)
ButtonC.configure(bg = "white")
ButtonCsharp.grid(row=3, column=0, columnspan=2)
ButtonD.grid(row=4, column=1)
ButtonD.configure(bg = "white")
ButtonDsharp.grid(row=3, column=1, columnspan=2)
ButtonE.grid(row=4, column=2)
ButtonE.configure(bg = "white")
ButtonF.grid(row=4, column=3)
ButtonF.configure(bg = "white")
ButtonFsharp.grid(row=3, column=3, columnspan=2)
ButtonG.grid(row=4, column=4)
ButtonG.configure(bg = "white")
ButtonGsharp.grid(row=3, column=4, columnspan=2)
ButtonA.grid(row=4, column=5)
ButtonA.configure(bg = "white")
ButtonAsharp.grid(row=3, column=5, columnspan=2)
ButtonB.grid(row=4, column=6)
ButtonB.configure(bg = "white")
ButtonC2.grid(row=4, column=7)
ButtonC2.configure(bg = "white")
ButtonClear.grid(row=4, column=8)


# Create Generate Signal button
GenerateWave = Button(root, text="Generate Sound Signal", padx=10, pady=20, command=lambda: signalMaker(note_list, 12, 4, int(sampleRate.get())))
GenerateWave.grid(row=5, column=0)

# Select signal type
wavestyle = StringVar()
waveoptions = OptionMenu(root, wavestyle,"Select Signal", "Additive Sin", "Additive Square", "Additive Sawtooth", "Wavetable")
wavestyle.set("Select Signal")
waveoptions.grid(row=5, column=2, padx=10, pady=10)

saveFile = Button(root, text="Save as WAV", padx=10, pady=20, command=lambda: save_as_wav("SunshineSynth.wav", signal, int(sampleRate.get()), amp=8000.0))
saveFile.grid(row=9, column=1, padx=10, pady=10)

# Creates Entry for note list display
e = Entry(root, width=200, borderwidth=5)
e.grid(row=0, column=0, columnspan=10, padx=10, pady=10)


# Current sample Rate display
SRlabel = Label(root, text="Current")
SRlabel.place(x=10, y=10)
SR2label = Label(root, text="Rate")
SR2label.place(x=10, y=25)
sampleRate = Entry(root, width=10, borderwidth=5)
sampleRate.insert(0, 48000)
sampleRate.grid(row=0, column=0, padx=10, pady=10)


# Filter Buttons

# Flanger Setup
flangerbutton = Button(root, text="flanger", padx=10, pady=20, command=lambda: flanger(signal, freq= 262, dry=float(dryFlange.get()), wet=float(wetFlange.get()), depth=20.0, delay=1.0, rate=int(sampleRate.get())))
flangerbutton.grid(row=6, column=0, padx=10, pady=10)
drylabel = Label(root, text="Dry")
drylabel.place(x=275, y=247)
dryFlange = Entry(root, width=10, borderwidth=5)
dryFlange.insert(0, 0.5)
dryFlange.grid(row=6, column=1, padx=10, pady=10)

wetlabel = Label(root, text="Wet")
wetlabel.place(x=475, y=247)
wetFlange = Entry(root, width=10, borderwidth=5)
wetFlange.insert(0, 0.5)
wetFlange.grid(row=6, column=2, padx=10, pady=10)

# Reverb setup
reverbbutton = Button(root, text="reverb", padx=10, pady=20, command=lambda: reverb(signal, echodur=float(revecho.get()), delay_amp=float(revdelay.get()), sample=sampleRate.get()))
reverbbutton.grid(row=6, column=3, padx=10, pady=10)

echolabel = Label(root, text="Echo")
echolabel.place(x=840, y=247)
revecho = Entry(root, width=10, borderwidth=5)
revecho.insert(0, 0.5)
revecho.grid(row=6, column=4, padx=10, pady=10)

echolabel = Label(root, text="Delay")
echolabel.place(x=1020, y=247)
revdelay = Entry(root, width=10, borderwidth=5)
revdelay.insert(0, 0.5)
revdelay.grid(row=6, column=5, padx=10, pady=10)

# distortion setup

distortionbutton = Button(root, text="distortion", padx=10, pady=20, command=lambda: distortion(signal,int(ampdist.get())))
distortionbutton.grid(row=7, column=0, padx=10, pady=10)
distlabel = Label(root, text="Amplification")
distlabel.place(x=275, y=330)
ampdist = Entry(root, width=10, borderwidth=5)
ampdist.insert(0, 10)
ampdist.grid(row=7, column=1, padx=10, pady=10)

# Tremolo setup
tremolobutton = Button(root, text="Tremolo", padx=10, pady=20, command=lambda: tremolo(signal, amp = int(tremamp.get()), speed = float(tremspeed.get())))
tremolobutton.grid(row=7, column=3, padx=10, pady=10)

amplabel = Label(root, text="Amplitude")
amplabel.place(x=840, y=330)
tremamp = Entry(root, width=10, borderwidth=5)
tremamp.insert(0, 1)
tremamp.grid(row=7, column=4, padx=10, pady=10)

amplabel = Label(root, text="Speed")
amplabel.place(x=1020, y=330)
tremspeed = Entry(root, width=10, borderwidth=5)
tremspeed.insert(0, 10)
tremspeed.grid(row=7, column=5, padx=10, pady=10)


# High pass filter
hpf = Button(root, text="High Pass Filter", padx=10, pady=20, command=lambda: butter_highpass_filter(signal, cutoff = float(hpfcutoff.get()), order = int(hpforder.get())))
hpf.grid(row=8, column=0, padx=10, pady=10)

hcutofflabel = Label(root, text="Cutoff")
hcutofflabel.place(x=275, y=415)
hpfcutoff = Entry(root, width=10, borderwidth=5)
hpfcutoff.insert(0, 500)
hpfcutoff.grid(row=8, column=1, padx=10, pady=10)

horderlabel = Label(root, text="Order")
horderlabel.place(x=470, y=415)
hpforder = Entry(root, width=10, borderwidth=5)
hpforder.insert(0, 1)
hpforder.grid(row=8, column=2, padx=10, pady=10)

# lowpass filter
lpf = Button(root, text="Low Pass Filter", padx=10, pady=20, command=lambda: lowpass(signal, cutoff = float(lpfcutoff.get()), order = int(lpforder.get())))
lpf.grid(row=8, column=3, padx=10, pady=10)

lcutofflabel = Label(root, text="Cutoff")
lcutofflabel.place(x=840, y=415)
lpfcutoff = Entry(root, width=10, borderwidth=5)
lpfcutoff.insert(0, 100)
lpfcutoff.grid(row=8, column=4, padx=10, pady=10)

lorderlabel = Label(root, text="Order")
lorderlabel.place(x=1020, y=415)
lpforder = Entry(root, width=10, borderwidth=5)
lpforder.insert(0, 1)
lpforder.grid(row=8, column=5, padx=10, pady=10)

# resample
resamplebut = Button(root, text="Resample", padx=10, pady=20, command=lambda: resampling(signal, int(newsr.get()), int(sampleRate.get())))
resamplebut.grid(row=8, column=6, padx=10, pady=10)
newsr = Entry(root, width=10, borderwidth=5)
newsr.insert(0, 32000)
newsr.grid(row=8, column=7, padx=10, pady=10)

bitbut = Button(root, text="Bit Depth", padx=10, pady=20, command=lambda: bitchange(signal))
bitbut.grid(row=7, column=6, padx=10, pady=10)
bitrate = StringVar()
bitoptions = OptionMenu(root, bitrate, "Select bitrate", "8 bit", "16 bit")
bitrate.set("Select bitrate")
bitoptions.grid(row=7, column=7, padx=10, pady=10)

# signal graphs
signalgraph = Button(root, text="Show graph", padx=10, pady=20, command=lambda: makegraph(signal))
signalgraph.grid(row=9, column=4, padx=10, pady=10)

fftgraph = Button(root, text="Show fft", padx=10, pady=20, command=lambda: makefft(signal))
fftgraph.grid(row=9, column=6, padx=10, pady=10)

effects = [flangerbutton, reverbbutton, distortionbutton, tremolobutton, hpf, lpf]
xwave = []



# Setups up wave table drawing

wavelabel = Label(root, text="WaveTable")  #labeling
wavelabel.place(x=40, y=560)
canvas = tk.Canvas(root, width=1600, height=400, highlightthickness=1, highlightbackground="black")
canvas.grid(row=10, column=0, columnspan=9, padx=10, pady=10)
canvas.old_coords = None
xpos = []
ypos = []

waveclearbutton = Button(root, text="Clear Wave", padx=10, pady=20, command=lambda: clearwave())
waveclearbutton.grid(row=9, column=7, padx=10, pady=10)

# Recognizes a user clicking in the box
def click(click_event):
    global prev
    prev = click_event

# starts tracking their mouse movements
def move(move_event):
    global prev
    canvas.create_line(prev.x, prev.y, move_event.x, move_event.y, width=2)  # takes the delta  x and y position, then store them as points for an array
    global xwave
    global ywave
    # if move_event.x > prev.x:
    xpos.append(move_event.x)
    ypos.append((move_event.y-200)/-400)  # normalizes the amplitude and shifts it to go from -1 to 1
    xwave = np.array(xpos)
    ywave = np.array(ypos)
    print('{}, {}'.format(move_event.x, move_event.y))
    prev = move_event

#  binds the buttons for each function
canvas.bind('<Button-1>', click)
canvas.bind('<B1-Motion>', move)

root.mainloop()

