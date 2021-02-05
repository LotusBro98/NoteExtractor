from pyffmpeg import FFmpeg
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np

import notes

SOURCE = "Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
TEMP = "temp.wav"


TIME_START = 0.5
TIME_END = 20
BPM = 110
FOURTHS_IN_BEAT = 3
MINOR_TICK_FOURTH = 1 / 2

FREQ_RES = 8
MAX_FREQ = notes.NOTES["C7"]
MIN_FREQ = notes.NOTES["C5"]

PLOT_RES = 1/6



BEAT_TIME = FOURTHS_IN_BEAT / (BPM / 60)
MINOR_TICK_BEAT = MINOR_TICK_FOURTH / FOURTHS_IN_BEAT

TIME_STEP_MAJOR = BEAT_TIME
TIME_STEP_MINOR = BEAT_TIME * MINOR_TICK_BEAT

TIME_RES = TIME_STEP_MINOR / 8
FREQ_RES_WINDOW_TIME = TIME_RES * 2


PLOT_RES_NOTE = PLOT_RES
PLOT_RES_BEAT = PLOT_RES

GRID_COLOR = 'gray'

def window_fourier(data, sample_rate):
    window_time = 1 / FREQ_RES
    window_samples = int(sample_rate * window_time)

    if TIME_END is not None:
        data = data[:int(sample_rate * TIME_END)]

    window_centers = np.linspace(0, len(data), int(len(data) / sample_rate / TIME_RES))
    data = np.pad(data, (window_samples, window_samples))
    window_centers = np.int32(window_centers) + window_samples
    windows = [data[center - window_samples // 2: center + window_samples // 2] for center in window_centers]
    windows = np.float32(windows)

    window_func = np.abs(np.linspace(-1, 1, windows.shape[-1]))
    window_func = np.clip(window_func * (1 / FREQ_RES) / FREQ_RES_WINDOW_TIME, 0, 1)
    window_func = 0.5 * (1 + np.cos(window_func * np.pi))

    windows *= window_func

    print(windows.shape)
    windows_fft = np.fft.fft(windows, axis=-1)
    max_freq_n = int(MAX_FREQ / FREQ_RES)
    min_freq_n = int(MIN_FREQ / FREQ_RES)
    windows_fft = windows_fft[:,:max_freq_n]
    windows_fft = np.abs(windows_fft)

    return windows_fft


def render_plot(spectrum):
    if TIME_END is None:
        max_time = spectrum.shape[0] * TIME_RES
    else:
        max_time = TIME_END

    notes_inside = dict([(note, notes.NOTES[note] / FREQ_RES) for note in notes.NOTES.keys() if
                         notes.NOTES[note] >= MIN_FREQ and notes.NOTES[note] <= MAX_FREQ])

    n_notes = len(notes_inside)

    im_size_x = PLOT_RES_BEAT * ((max_time - TIME_START) / TIME_STEP_MINOR)
    im_size_y = PLOT_RES_NOTE * n_notes

    padding_x = 9
    padding_y = 5

    figsize_x = im_size_x + padding_x
    figsize_y = im_size_y + padding_y

    print(figsize_x, figsize_y)

    fig = plt.figure(figsize=(figsize_x, figsize_y), tight_layout = {'pad': 3})
    ax = fig.add_subplot(111)

    ax.set_yscale('log')
    max_freq_n = int(MAX_FREQ / FREQ_RES)
    min_freq_n = int(MIN_FREQ / FREQ_RES)
    ax.set_ylim((min_freq_n, max_freq_n))

    start_time_n = TIME_START / TIME_RES
    max_time_n = max_time / TIME_RES
    ax.set_xlim((start_time_n, max_time_n))

    aspect1 = ((max_time - TIME_START) / TIME_RES) / (np.log10(MAX_FREQ / FREQ_RES) - np.log10(MIN_FREQ / FREQ_RES))
    aspect = aspect1 * im_size_y / im_size_x

    xticks_major = np.arange(TIME_START, max_time, TIME_STEP_MAJOR) / TIME_RES
    xticks_minor = np.arange(TIME_START, max_time, TIME_STEP_MINOR) / TIME_RES
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(np.arange(1, len(xticks_major) + 1, 1))

    ax.xaxis.grid(True, which='minor', color=GRID_COLOR, linestyle="--", alpha=0.3)
    ax.xaxis.grid(True, which='major', color=GRID_COLOR, linewidth=2, alpha=0.5)

    ax.set_yticks(list(notes_inside.values()), minor=True)
    ax.set_yticklabels(list(notes_inside.keys()), minor=True)

    ax.yaxis.grid(True, which='minor', color=GRID_COLOR, linestyle="--", alpha=0.3)

    # ax.set_aspect(aspect)

    ax.imshow(spectrum.T, aspect=aspect)

    plt.show()


if __name__ == '__main__':
    ff = FFmpeg()
    ff.convert(SOURCE, TEMP)

    sample_rate, track = read(TEMP)
    track = track[:,0]
    track = track / 2**15

    spectrum = window_fourier(track, sample_rate)

    render_plot(spectrum)
