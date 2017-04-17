#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.savefig("train.png")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#print(features_train)


from sklearn.metrics import accuracy_score
from time import time

################
# SVM
###############

#from sklearn.svm import SVC
#c_value = 10000.
#clf = SVC(kernel = 'rbf', C=c_value)
#clf = SVC(kernel = 'linear')

################
# Decision Tree
###############

min_samples_split = 2
criterion = 'gini'

print("Parameters: min_split={}, criterion={}".format(min_samples_split, criterion))

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split = min_samples_split)

################
# Random Forest
###############

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=None, criterion=criterion, min_samples_split = min_samples_split, n_estimators=10)


t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" 

t0 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s" 

accuracy = accuracy_score(labels_test, pred)

print("Accuracy: {}".format(accuracy))





try:
    prettyPicture(clf, features_test, labels_test)
    #plt.show()
except NameError:
    pass
