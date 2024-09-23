import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter, butter, lfilter
import seaborn as sns


fs = 200.0


df = pd.read_csv('file1.csv')

N = len(df)

df['time'] = np.arange(N) / fs 
time = df['time'].values
signal = df['column_y'].values

window = np.hanning(N)
windowed_signal = signal * window

def butterworth_filter(signal, cutoff, fs, order=20):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False )
    return lfilter(b, a, signal)

plt.figure(figsize=(12, 9))


plt.subplot(5, 1, 1)
sns.lineplot(x = time, y = windowed_signal, color='blue', label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')


window_length = 51  
polyorder = 3  
signal_smooth = savgol_filter(windowed_signal, window_length=window_length, polyorder=polyorder)

# Plot the smoothed signal
plt.subplot(5, 1, 2)
sns.lineplot(x = time, y = signal_smooth, color='orange', label='Smoothed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Smoothed Signal (Savitzky-Golay Filter)')


cutoff_frequency = 40.0
filtered_signal = butterworth_filter(windowed_signal, cutoff=cutoff_frequency, fs=fs)


yf_filtered = fft(filtered_signal)
xf = fftfreq(N, 1/fs)


plt.subplot(5, 1, 3)
sns.lineplot(x= xf[:N//2], y = fft(signal_smooth)[:N//2] / N, color='green')  # Normalize FFT
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Frequency (Smoothed Signal)')


plt.subplot(5, 1, 4)
sns.lineplot(x = xf[:N//2], y = fft(windowed_signal)[:N//2] / N, color='black')  # Normalize FFT
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Frequency (Non-Filtered Signal)')


plt.subplot(5, 1, 5)
sns.lineplot(x = xf[:N//2], y = yf_filtered[:N//2] /(2* N), color='magenta')  # Normalize FFT
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Frequency (Filtered Signal)')




plt.tight_layout()
plt.show()






'''
order - 20

file 1 - 11
file 2 - 
file 3
file 4
file 5
'''