
  
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import scipy 



# X,Y,X Mean
def calculate_mean(window):
    return np.mean(window, axis=0)



# X,Y,X Standard Deviation
def calculate_std(window):
    return np.std(window, axis=0)



# Standard Deviation of Magnitude
def std_magnitude(window):
    return [np.std(get_magnitude(window))]



# Magnitude
def get_magnitude(window):

    magnitude = np.zeros(len(window))
    for i in range(len(window)):
        temp = (window[i][0]**2 + window[i][1]**2 + window[i][2]**2) ** 0.5
        magnitude[i] = temp

    return magnitude



# Mean of Magnitude
def mean_magnitude(window):
    magnitude = get_magnitude(window)
    return [np.mean(magnitude)]


# X,Y,X Variance
def calculate_variance(window):
    return np.var(window, axis=0)


# FFT
def fft_feature_calculator(window):
    fft_arr = np.fft.rfft(get_magnitude(window))
    return [sum(fft_arr.astype(float))] # could also experiment with max()




# Peak Count
def count_peaks(window):
    magnitude = get_magnitude(window)
    peaks = signal.find_peaks(magnitude)[0]
    
    return [len(peaks)]



# Mean Peak Height
def mean_peak_height(window):
    magnitude = get_magnitude(window)
    peaks = signal.find_peaks(magnitude)[0]
    
    if len(peaks) == 0:
        return [0]
    
    return [np.mean(peaks)]




# Entropy
def entropy_calculator(window):
    mag = get_magnitude(window)
    counts,bins = np.histogram(mag,20)
    counts+=1
    PA = counts / np.sum(counts, dtype=float)
    SA = -PA * np.log(PA)
    return [-np.sum(PA * np.log(PA),axis=0)]



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

    x.append(fft_feature_calculator(window))
    feature_names.append("fft")


    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector

    return feature_names, feature_vector
