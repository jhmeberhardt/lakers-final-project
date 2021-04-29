import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as pl
from scipy.signal import butter, freqz, filtfilt, firwin, lfilter

# Parsing accelerometer data
def pull_data(file_name):
    f = open(file_name + '.csv')
    xs = []
    timestamps = []
    for line in f:
        value = line.split(',')
        #print(value[0])
        #zyx
        timestamps.append(float(value[1]))
        # r = 0
        # for i in range(2, len(value)):
        #     r += float(value[i]) ** 2
        # rs.append(r)
        xs.append(value[len(value)-1])
    return np.array(xs), np.array(timestamps)



#pulling data
accel_file = 'jack_pushups'
signal, timestamps = pull_data(accel_file)



#plotting raw data
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'r-',label='unfiltered')
plt.title("Unfiltered Pushups Signal")
pl.legend(loc='upper left')
plt.xlabel("Time (s)")
plt.ylabel("Height (10^-3 m)")
plt.grid()
plt.show()



# low-pass filter
order = 3
fs = 10
cutoff = 2  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPRIATE
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
lp_b, lp_a = butter(order, normal_cutoff, btype='low', analog=False)
lp_signal = filtfilt(lp_b, lp_a, signal)
#signal = [x - 97 for x in signal]
# signal = signal - np.mean(signal)


plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'r-', label = 'unfiltered')
plt.plot(timestamps, lp_signal, 'b-', label = 'processed')
plt.plot(timestamps, signal, 'r-',label='unfiltered')



plt.title("Filtered data vs Unfiltered data")
pl.legend(loc='upper left')
plt.xlabel("Time (s)")
plt.ylabel("Height (10^-3 m)")
plt.grid()
plt.show()
