
  
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
from scipy.signal import find_peaks

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features


def get_magnitude(window):
    '''
    Not being used right now
    '''
    magnitude = np.zeros(len(window))

    for i in range(len(window)):
        temp = (window[i][0]**2 + window[i][1]**2 + window[i][2]**2) ** 0.5
        magnitude[i] = temp

    magnitude_mean = np.mean(magnitude)
    return magnitude

def calculate_variance(window):
    return np.var(window, axis=0)

def fft_feature_calculator(window):
    fft_arr = np.fft.rfft(get_magnitude(window))
    return [sum(fft_arr.astype(float))] # could also experiment with max()


def entropy_calculator(window):
    mag = get_magnitude(window)
    counts,bins = np.histogram(mag,20)
    counts+=1

    PA = counts / np.sum(counts, dtype=float)
    SA = -PA * np.log(PA)
    return [-np.sum(PA * np.log(PA),axis=0)]

def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature vector.
    """

    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    x.append(calculate_variance(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")

    x.append(fft_feature_calculator(window))
    feature_names.append("Magnitude Fourier Transform")

    x.append(count_peaks_feature(window))
    feature_names.append("Magnitude peak count")

    x.append(entropy_calculator(window))
    feature_names.append("Entropy")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector

    return feature_names, feature_vector
