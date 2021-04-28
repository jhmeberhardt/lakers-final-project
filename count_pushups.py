import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as pl
from scipy.signal import butter, freqz, filtfilt, firwin, lfilter

# Parsing accelerometer data
def pull_data(file_name):
    f = open(file_name + '.csv')
    rs = []
    timestamps = []
    for line in f:
        value = line.split(',')
        #print(value[0])
        #zyx
        timestamps.append(float(value[1]))
        r = 0
        for i in range(2, len(value)):
            r += float(value[i]) ** 2
        rs.append(r)
    return np.array(rs), np.array(timestamps)



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



# high-pass filter
order = 5
fs = 100
cutoff = 10  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPRIATE
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
hp_b, hp_a = butter(order, normal_cutoff, btype='high', analog=False)
hp_signal = filtfilt(hp_b, hp_a, signal)
#signal = [x - 97 for x in signal]


plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'r-', label = 'unfiltered')
plt.plot(timestamps, hp_signal, 'b-', label = 'processed')



plt.title("Filtered data vs Unfiltered data")
pl.legend(loc='upper left')
plt.xlabel("Time (s)")
plt.ylabel("Height (10^-3 m)")
plt.grid()
plt.show()
