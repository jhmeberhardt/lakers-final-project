import os
import sys
import numpy as np

# import sklearn
import extract_data
# from sklearn.tree import export_graphviz
# from sklearn import model_selection
# from sklearn import tree
from features import extract_features
from util import slidingWindow, reorient, reset_vars
# import pickle
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")    
sys.stdout.flush()
data_file = 'jack_pushups.csv'
data = np.genfromtxt(data_file, comments="#", delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()


# ---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,4], data[i,3], data[i,2]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,1:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)



# ---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds


class_names = ["standing up", "pushups", "laying down"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])



X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()