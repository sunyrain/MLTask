import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_path = 'Dataset/train/AddWeight/AddWeight_0.csv'
df = pd.read_csv(input_path)

#time_column = df.iloc[:, 0]
data_columns = df.iloc[:, 1:]

time = np.linspace(0, len(df), len(df))
# Choose a specific data column for FFT (you can adjust this)

voltage = np.array(df.iloc[:, 7])
# Perform FFT
fft_freq = np.fft.fftfreq(len(time))
fft_result = np.fft.fft(voltage)
fft_result_shifted = np.fft.fftshift(fft_result)
fft_freq_shifted = np.fft.fftshift(fft_freq)

plt.figure(figsize=(10, 6))
plt.plot(time, voltage)
plt.title('')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Plot the FFT result
plt.figure(figsize=(10, 6))
plt.plot(fft_freq_shifted, np.abs(fft_result_shifted))
plt.title('FFT Result with fftshift')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()