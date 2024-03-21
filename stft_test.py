from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#data
fs = 48000
T = 1
t = np.arange(0, int(T*fs)) / fs
w = signal.chirp(t, f0=20, f1=20000, t1=T, method='logarithmic')

f, t, Zxx = signal.stft(w, fs, nperseg=512)
amp = np.abs(Zxx).max()  # You might want to dynamically set vmax for better visualization

# Plotting

pcm = plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
cbar = plt.colorbar(pcm)
cbar.set_label('Amplitude')
plt.show()