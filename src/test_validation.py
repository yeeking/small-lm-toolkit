import shared_utils
from pathlib import Path
import pretty_midi
import pypianoroll
import matplotlib.pyplot as plt
import torch 
import os 
import librosa

def midi_to_pretty_midi(midifile:str):
    """ load sent midi file and convert to pretty_midi object """
    assert os.path.exists(midifile), f"Cannot find midi file {midifile}"
    pm = pretty_midi.PrettyMIDI(midifile)
    return pm 

def pretty_midi_to_fig(pm:pretty_midi.PrettyMIDI):
    """convert sent pretty midi object into a pyplot fig"""
    # your code to plot the piano roll, as we discussed
    fig = plt.figure(figsize=(8,3))
    # mt = pypianoroll.from_pretty_midi(pm)
    # pypianoroll.plot(mt, preset="full")
    fig.tight_layout()
    fs = 100 # 1/fs steps per second
    start_pitch = 24
    end_pitch = 96
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

    return fig



def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


def pretty_midi_to_audio(pm:pretty_midi.PrettyMIDI, sr:int=16000):
    """render sent pretty_midi object into wonderful music and return raw audio samples"""
    audio = pm.fluidsynth(fs=sr)  # numpy [T]
    wave = torch.from_numpy(audio).unsqueeze(0)  # [1, T], float
    return wave, sr


midifile = 'jazz.mid'
pm = midi_to_pretty_midi(midifile)

# fig = plt.figure(figsize=(8,3))
# fig.tight_layout()
# plot_piano_roll(pm, 24, 84)

fig = pretty_midi_to_fig(pm)
fig.savefig('music.png')
