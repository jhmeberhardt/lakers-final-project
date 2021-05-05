import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import extract_data
from sklearn.tree import export_graphviz
from sklearn import model_selection
from sklearn import tree
from features import extract_features, get_magnitude
from util import slidingWindow, reorient, reset_vars
import pickle
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/jack_pushups.csv'
data = np.genfromtxt(data_file, comments="#", delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()


# -----------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
print("Reorienting accelerometer data...")
data_with_timestamps = []
for i in range(0,len(data)):
    data_with_timestamps.append(np.flip(data[i]))
data_with_timestamps = np.array(data_with_timestamps)
sys.stdout.flush()
reset_vars()

# -----------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 50
step_size = 10

class_names = ["standing up", "pushups", "laying down","walking","jumping jacks","running in place"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
count = 0
for i,window_with_timestamp_and_label in slidingWindow(data_with_timestamps, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]
    feature_names, x = extract_features(window)
    X.append(x)

    # Labeling data
    if count < 8:
        Y.append("laying down")
    elif count < 62:
        Y.append("pushups")
    elif count < 71:
        Y.append("standing up")
    elif count < 139:
        Y.append("walking")
    elif count < 201:
        Y.append("jumping jacks")
    elif count < 259:
        Y.append("running")
    else:
        Y.append("pushups")
    count+=1


X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()




# -----------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# Cross validation
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)


accuracies = []
precisions = []
recalls = []
f1_scores = []
i = 1
for train, test, in cv.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    tree.fit(X_train,y_train)
    prediction = tree.predict(X_test)
    conf = confusion_matrix(y_test, prediction)

    accuracy = accuracy_score(y_test,prediction)
    precision = precision_score(y_test,prediction,average='weighted')
    recall = recall_score(y_test,prediction,average='weighted')
    f1 = f1_score(y_test,prediction,average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print("Fold " + str(i) + ": \nAccuracy: "+ str(accuracy) + ", \nPrecision: " + str(precision) + ", \nRecall: " + str(recall) + ", \nConfusion Matrix: \n" + str(conf) + ", \nF1 Score: " + str(f1) + "\n\n\n")
    i+=1


# Computing averages
avg_accuracy = 0
for num in accuracies:
    avg_accuracy += num

avg_accuracy = avg_accuracy / len(accuracies)

avg_precision = 0
for num in precisions:
    avg_precision += num

avg_precision = avg_precision / len(precisions)

avg_recall = 0
for num in recalls:
    avg_recall += num

avg_recall = avg_recall / len(recalls)

avg_f1 = 0
for num in f1_scores:
    avg_f1 += num

avg_f1 = avg_f1 / len(f1_scores)


print("Average Accuracy: %s\nAverage Precision: %s\nAverage Recall: %s\nAverage F1: %s" % (avg_accuracy,avg_precision,avg_recall,avg_f1))



