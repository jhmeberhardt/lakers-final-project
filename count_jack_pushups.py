import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as pl
from scipy.signal import butter, freqz, filtfilt, firwin, lfilter
import extract_data

# Parsing accelerometer data (iphon



#pulling data
accel_file = 'jack_pushups'
signal, timestamps = extract_data.pull_data_iphone(accel_file)



#plotting raw data

# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'r-',label='unfiltered')
# plt.title("Unfiltered Pushups Signal")
# pl.legend(loc='upper left')
# plt.xlabel("Time (s)")
# plt.ylabel("Height (10^-3 m)")
# plt.grid()
# plt.show()



# low-pass filter
order = 5
fs = 50
cutoff = 1  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPRIATE
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
lp_b, lp_a = butter(order, normal_cutoff, btype='low', analog=False)
lp_signal = filtfilt(lp_b, lp_a, signal)
#signal = [x - 97 for x in signal]
# signal = signal - np.mean(signal)


lp_signal = lp_signal[800:2400]
timestamps = timestamps[800:2400]

pushups = []
# threshold = 0
for i in range (1, len(lp_signal) - 1):
    cur = lp_signal[i]
    if (cur - (-2) < 0.001 and lp_signal[i+1] > -2):
        pushups.append(i)
print(pushups)






plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'r-', label = 'unfiltered')
plt.plot(timestamps, lp_signal, 'b-', label = 'filtered data')
#plt.plot(timestamps, signal, 'r-',label='unfiltered')
for index in pushups:
    pl.plot(timestamps[index], lp_signal[index], 'x', color='black')

plt.title("Filtered pushup data with pushups marked")
pl.legend(loc='upper left')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/(s^2))")
plt.grid()
plt.show()

# Now that data filtered, graphed, counted -- process:


    

















