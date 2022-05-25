"""
Synthetic Dataset 2
Name: Haolun Cheng
USCID: 1882563827
EE559 HW1
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from plotDecBoundaries import plotDecBoundaries

clas1xtotal = []
clas1ytotal = []
clas2xtotal = []
clas2ytotal = []
allclassdatapoints = []
allclasslabels = []
allclassmeans = []

# Open train csv file for training the classifier
with open('synthetic2_train.csv', 'r') as train:
    training_set = csv.reader(train)

    # Train
    for line in training_set:
        x, y, label = line[0], line[1], line[2]
        allclassdatapoints.append((float(x), float(y)))
        allclasslabels.append(label)
        if label == '1':
            clas1xtotal.append(float(x))
            clas1ytotal.append(float(y))
        elif label == '2':
            clas2xtotal.append(float(x))
            clas2ytotal.append(float(y))

train.close()

# Compute mean for each class
clas1xmean = np.mean(clas1xtotal)
clas1ymean = np.mean(clas1ytotal)

clas2xmean = np.mean(clas2xtotal)
clas2ymean = np.mean(clas2ytotal)

clas1_mean_point = np.array((clas1xmean, clas1ymean))
clas2_mean_point = np.array((clas2xmean, clas2ymean))

# Variables for plotDecBoundaries
allclassmeans = [[clas1xmean, clas1ymean], [clas2xmean, clas2ymean]]
datapoints = np.array(allclassdatapoints).astype(float)
claslabels = np.array(allclasslabels).astype(float)
samplemeans = np.array(allclassmeans).astype(float)

# Classify data points (training set)
countTrainingError = 0
totalTrainingPoints = 0
with open('synthetic2_train.csv', 'r') as training:
    train_set = csv.reader(training)

    for line in train_set:
        totalTrainingPoints += 1
        x, y, label = line[0], line[1], line[2]
        trainPoint = np.array((x, y))
        dist1 = np.linalg.norm(trainPoint.astype(float) - clas1_mean_point)
        dist2 = np.linalg.norm(trainPoint.astype(float) - clas2_mean_point)

        if dist1 < dist2:
            if label != '1':
                countTrainingError += 1
        elif dist1 > dist2:
            if label != '2':
                countTrainingError += 1

training.close()

# Fine error rate for training set
error_rate = float(countTrainingError) / float(totalTrainingPoints)
print("Error rate for the training set: " + str(error_rate))

# Classify data points (test set)
countTestError = 0
totalTestPoints = 0
with open('synthetic2_test.csv', 'r') as test:
    test_set = csv.reader(test)

    for line in test_set:
        totalTestPoints += 1
        x, y, label = line[0], line[1], line[2]
        testPoint = np.array((x, y))
        dist1 = np.linalg.norm(testPoint.astype(float) - clas1_mean_point)
        dist2 = np.linalg.norm(testPoint.astype(float) - clas2_mean_point)

        if dist1 < dist2:
            if label != '1':
                countTestError += 1
        elif dist1 > dist2:
            if label != '2':
                countTestError += 1

test.close()

# Fine error rate for test set
error_rate = float(countTestError) / float(totalTestPoints)
print("Error rate for the test set: " + str(error_rate))

# Plot the data points
xAxis = [i[0] for i in allclassdatapoints]
yAxis = [j[1] for j in allclassdatapoints]
plt.plot(xAxis, yAxis, 'r.')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Feature Plot of all Elements')
plt.show()

# Plot the decision boundaries
plotDecBoundaries(datapoints, claslabels, samplemeans)