"""
Wine Dataset (for question (c))
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
clas3xtotal = []
clas3ytotal = []
allclasstwofeatures = []
allclasslabels = []
allclassmeans = []
feature1 = []
feature2 = []
feature3 = []

# Open train csv file for training the classifier
with open('wine_train.csv', 'r') as train:
    training_set = csv.reader(train)

    # Train
    for line in training_set:
        x, y, label = line[0], line[1], line[-1]
        allclasstwofeatures.append((float(x), float(y)))
        allclasslabels.append(label)
        if label == '1':
            clas1xtotal.append(float(x))
            clas1ytotal.append(float(y))
            feature1.append((float(x), float(y)))
        elif label == '2':
            clas2xtotal.append(float(x))
            clas2ytotal.append(float(y))
            feature2.append((float(x), float(y)))
        else:
            clas3xtotal.append(float(x))
            clas3ytotal.append(float(y))
            feature3.append((float(x), float(y)))

train.close()

# Compute mean for each class
clas1xmean = np.mean(clas1xtotal)
clas1ymean = np.mean(clas1ytotal)

clas2xmean = np.mean(clas2xtotal)
clas2ymean = np.mean(clas2ytotal)

clas3xmean = np.mean(clas3xtotal)
clas3ymean = np.mean(clas3ytotal)

clas1_mean_point = np.array((clas1xmean, clas1ymean))
clas2_mean_point = np.array((clas2xmean, clas2ymean))
clas3_mean_point = np.array((clas3xmean, clas3ymean))

# Variables for plotDecBoundaries
allclassmeans = [[clas1xmean, clas1ymean], [clas2xmean, clas2ymean], [clas3xmean, clas3ymean]]
twofeatures = np.array(allclasstwofeatures).astype(float)
claslabels = np.array(allclasslabels).astype(float)
samplemeans = np.array(allclassmeans).astype(float)

# Classify data points (training set)
countTrainingError = 0
totalTrainingPoints = 0
with open('wine_train.csv', 'r') as training:
    train_set = csv.reader(training)

    for line in train_set:
        totalTrainingPoints += 1
        x, y, label = line[0], line[1], line[-1]
        trainPoint = np.array((x, y))
        dist1 = np.linalg.norm(trainPoint.astype(float) - clas1_mean_point)
        dist2 = np.linalg.norm(trainPoint.astype(float) - clas2_mean_point)
        dist3 = np.linalg.norm(trainPoint.astype(float) - clas3_mean_point)

        if dist1 < dist2 and dist1 < dist3:
            if label != '1':
                countTrainingError += 1
        if dist2 < dist1 and dist2 < dist3:
            if label != '2':
                countTrainingError += 1
        if dist3 < dist1 and dist3 < dist2:
            if label != '3':
                countTrainingError += 1

training.close()

# Fine error rate for training set
error_rate = float(countTrainingError) / float(totalTrainingPoints)
print("Error rate for the training set: " + str(error_rate))

# Classify data points (test set)
countTestError = 0
totalTestPoints = 0
with open('wine_test.csv', 'r') as test:
    test_set = csv.reader(test)

    for line in test_set:
        totalTestPoints += 1
        x, y, label = line[0], line[1], line[-1]
        testPoint = np.array((x, y))
        dist1 = np.linalg.norm(testPoint.astype(float) - clas1_mean_point)
        dist2 = np.linalg.norm(testPoint.astype(float) - clas2_mean_point)
        dist3 = np.linalg.norm(testPoint.astype(float) - clas3_mean_point)

        if dist1 < dist2 and dist1 < dist3:
            if label != '1':
                countTestError += 1
        if dist2 < dist1 and dist2 < dist3:
            if label != '2':
                countTestError += 1
        if dist3 < dist1 and dist3 < dist2:
            if label != '3':
                countTestError += 1

test.close()

# Fine error rate for test set
error_rate = float(countTestError) / float(totalTestPoints)
print("Error rate for the test set: " + str(error_rate))

# Plot the data points
xAxis1 = [i[0] for i in feature1]
yAxis1 = [i[1] for i in feature1]
xAxis2 = [i[0] for i in feature2]
yAxis2 = [i[1] for i in feature2]
xAxis3 = [i[0] for i in feature3]
yAxis3 = [i[1] for i in feature3]
plt.plot(xAxis1, yAxis1, 'r.', xAxis2, yAxis2, 'b^', xAxis3, yAxis3, 'g.')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Feature Plot of all Elements')
plt.show()

# Plot the decision boundaries
plotDecBoundaries(twofeatures, claslabels, samplemeans)