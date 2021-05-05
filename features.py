
  
# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.
As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import scipy 



# ---------------------------------------------------------------------------
#		                    Calculate Mean X,Y,Z
# -----------------------------------------------------------------------------
def calculate_mean(window):
    return np.mean(window, axis=0)




# ---------------------------------------------------------------------------
#		                    Calculate Standard Deviation X,Y,Z
# -----------------------------------------------------------------------------
def calculate_std(window):
    return np.std(window, axis=0)



# ---------------------------------------------------------------------------
#		                    Calculate StdDev of Magnitude
# -----------------------------------------------------------------------------
def std_magnitude(window):
    return [np.std(get_magnitude(window))]




# ---------------------------------------------------------------------------
#		                    Calculate Magnitude
# -----------------------------------------------------------------------------
def get_magnitude(window):

    magnitude = np.zeros(len(window))

    for i in range(len(window)):
        temp = (window[i][0]**2 + window[i][1]**2 + window[i][2]**2) ** 0.5
        magnitude[i] = temp

    
    return magnitude



# ---------------------------------------------------------------------------
#		                    Calculate Mean of Magnitude
# -----------------------------------------------------------------------------
def mean_magnitude(window):
    magnitude = get_magnitude(window)
    return [np.mean(magnitude)]


# ---------------------------------------------------------------------------
#		                    Calculate Variance
# -----------------------------------------------------------------------------
def calculate_variance(window):
    return np.var(window, axis=0)


# ---------------------------------------------------------------------------
#		                    FFT
# -----------------------------------------------------------------------------
def fft_feature_calculator(window):
    fft_arr = np.fft.rfft(get_magnitude(window))
    return [sum(fft_arr.astype(float))] # could also experiment with max()




# ---------------------------------------------------------------------------
#		                    Count Peaks
# -----------------------------------------------------------------------------
def count_peaks(window):
    magnitude = get_magnitude(window)
    peaks = signal.find_peaks(magnitude)[0]
    
    return [len(peaks)]



# ---------------------------------------------------------------------------
#		                    Mean Peak Height
# -----------------------------------------------------------------------------
def mean_peak_height(window):
    magnitude = get_magnitude(window)
    peaks = signal.find_peaks(magnitude)[0]
    
    if len(peaks) == 0:
        return [0]
    

    return [np.mean(peaks)]




# ---------------------------------------------------------------------------
#		                    Entropy
# -----------------------------------------------------------------------------
def entropy_calculator(window):
    mag = get_magnitude(window)
    counts,bins = np.histogram(mag,20)
    counts+=1

    PA = counts / np.sum(counts, dtype=float)
    SA = -PA * np.log(PA)
    return [-np.sum(PA * np.log(PA),axis=0)]




# helper function
# def find_peaks(window):
#     #peak_timestamps = []
#     peaks = []
#     for i in range(1, len(window)-1):
#         if (window[i] > window[i-1]) and (window[i] > window[i+1]):
#             #step_timestamps.append(timestamps[i])
#             peaks.append(window[i])
#     return peaks


def extract_features(window):


    x = []
    feature_names = []



    x.append(calculate_mean(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    x.append(calculate_variance(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")

    x.append(calculate_std(window))
    feature_names.append("x_std")
    feature_names.append("y_std")
    feature_names.append("z_std")

    x.append(std_magnitude(window))
    feature_names.append("magnitude_std")

    x.append(mean_magnitude(window))
    feature_names.append("magnitude_mean")


    x.append(count_peaks(window))
    feature_names.append("magnitude peak count")

    x.append(mean_peak_height(window))
    feature_names.append("mean peak height")

    x.append(entropy_calculator(window))
    feature_names.append("entropy")

    



    # print(count_peaks(window))
    # print(get_magnitude(window))
    #print(window)

    # x.append(mean_peak_height(window))
    # feature_names.append("mean peak height of magnitude")


    # x.append(mean_peak_distance(window))
    # feature_names.append("mean peak distance of magnitude")


    # for i in range(0,len(x)):
    #     print(str(x[i]) + "\n\n")


    # plt.figure(figsize=(10,5))
    # plt.plot(get_magnitude(window), 'b-', label = 'filtered data')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (m/(s^2))")
    # plt.grid()
    # plt.show()

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector

    return feature_names, feature_vector
