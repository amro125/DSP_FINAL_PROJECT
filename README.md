# DSP_FINAL_PROJECT
## Amit Rogel and Lauren McCall


## How to use:
 Run the main file script. This will open a new gui window. That is where all of the editing will be done. The synthesizer is modular, but has a linear order of operations.
## 1) Create the sound signal
This wcan be done by assigning note and using either a sin, square, or sawtooth wave.  In addition, you can draw a wave in the bottom box for a wavetable synthesizer.

## 2) Apply effects

Effects can be applied by clicking on the button. The button will turn green once an effect is applied. You can set them in any order you would like

## 3) Reclicking "generate signal" will erase the effects

If you need to undo and start your effects over, clikc the generate sound signal button. Clicking clear in the top will erase the note progression. Selecting the clear wave erases the drawn wave

## 4) View the sound

You can show a graph of the sound signal or an fft plot that will display phase and magnitude

## 5) Exporting sound file

You can downsample you file by setting the new sample rate and hitting "resample." You can also create a lower bit depth with the bit depth button. If you would like to export your file normally, click the "save as wav" button. This will export your file as a wav file under the name "sunshinesynth.wav"
