from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

#Load data
iris = load_iris() 
X = iris.data      
y = iris.target

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)

#Define and fit to the model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(accuracy_score(predicted, y_test))
print(clf.predict(X_test))

os.chdir("D:/MSC/00_Project/07_IRIS/code")
with open(r'rf.pkl','wb') as model_pkl:
    pickle.dump(clf, model_pkl, protocol=2)
