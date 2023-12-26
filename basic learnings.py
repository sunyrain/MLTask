import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
print(sys.path)
list = [1200, 1200, 1200, 1200, 0.5, 700, 12, 15, 10, 200, 20, 10, 10, 10, 30,
        40]  # Maximum value for each sensor set based on experience


def Normal(Data):
    Data = Data.astype(np.float32)
    for id in range(16):
        Data[:, id] = Data[:, id] / list[id]
    return Data


Xtrain = np.zeros([975, 180, 16])
Ytrain = np.zeros([975])
t = 0
for i in range(214):
    T_Data = pd.read_csv('Dataset/train/AddWeight/AddWeight_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 1
        t = t+1
for i in range(146):
    T_Data = pd.read_csv('Dataset/train/Normal/Normal_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 2
        t = t+1
for i in range(213):
    T_Data = pd.read_csv('Dataset/train/PressureGain_constant/PressureGain_constant_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 3
        t = t+1
for i in range(199):
    T_Data = pd.read_csv('Dataset/train/PropellerDamage_bad/PropellerDamage_bad_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 4
        t = t+1
for i in range(208):
    T_Data = pd.read_csv('Dataset/train/PropellerDamage_slight/PropellerDamage_slight_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 5
        t = t+1
Xtrain1 = np.zeros([975, 180*16])
for i in range(975):
    for j in range(180):
        for k in range(16):
            Xtrain1[i][j*16+k] = Xtrain[i][j][k]

Xtest = np.zeros([245, 180, 16])
Ytest = np.zeros([245])
t = 0
for i in range(54):
    T_Data = pd.read_csv('Dataset/test/AddWeight/AddWeight_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 1
        t = t+1
for i in range(36):
    T_Data = pd.read_csv('Dataset/test/Normal/Normal_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 2
        t = t+1
for i in range(53):
    T_Data = pd.read_csv('Dataset/test/PressureGain_constant/PressureGain_constant_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 3
        t = t+1
for i in range(50):
    T_Data = pd.read_csv('Dataset/test/PropellerDamage_bad/PropellerDamage_bad_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 4
        t = t+1
for i in range(52):
    T_Data = pd.read_csv('Dataset/test/PropellerDamage_slight/PropellerDamage_slight_'+str(i)+'.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 5
        t = t+1
Xtest1 = np.zeros([245, 180*16])
for i in range(245):
    for j in range(180):
        for k in range(16):
            Xtest1[i][j*16+k] = Xtest[i][j][k]

clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0, n_estimators=1000)
clf = clf.fit(Xtrain1, Ytrain)
rfc = rfc.fit(Xtrain1, Ytrain)
score_c = clf.score(Xtest1, Ytest)
score_r = rfc.score(Xtest1, Ytest)
print("Single Tree: {}" .format(score_c))
print("Random Forest: {}" .format(score_r))

k = ['linear', 'poly', 'rbf', 'sigmoid']
svm_classification = SVC(C=5.0, kernel=k[1], degree=5)
svm_classification = svm_classification.fit(Xtrain1, Ytrain)
score_s = svm_classification.score(Xtest1, Ytest)
print("Support Vector Machine: {}" .format(score_s))

bag = BaggingClassifier()
bag = bag.fit(Xtrain1, Ytrain)
score_b = bag.score(Xtest1, Ytest)
print("Bagging: {}" .format(score_b))