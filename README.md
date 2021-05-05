# lakers-final-project
Jack Eberhardt, Sankalp Ramesh, Benjamin Xu
Final Project for CS328


Our idea for our final project has to do with pushups. Pushups are a common and fairly easy bodyweight exercise that can be performed by most anyone for general fitness and working out their chest. What we would do would be to recognize pushups being done based on analysis of sensor data, count the amount of pushups done (using a certain standardized measurement program designed by us) in a specific amount of time. We would could the amount of pushups performed and suggest to the user a realistic exercise goal to increase the amount of pushups they can do in general. 

What we ended up doing was training a classifier and detecting different exercises and distinguishing between them. 

Classification and prediction is done in classification.py, feature extraction is done in features.py, and our data is in the data folder. 

We recorded multiple different isolated sessions of exercises and tasks, and concatenated them in all_data.csv. This has the multiple minutes of recorded data inside of it, with the end of each section marked with a comment. 

We also used a few functions from util.py (from assignment 2 part 1) like slidingWindow. 












