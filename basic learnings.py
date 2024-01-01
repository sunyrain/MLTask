#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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
    T_Data = pd.read_csv('./train/AddWeight/AddWeight_' + str(i) + '.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 1
        t = t + 1
for i in range(146):
    T_Data = pd.read_csv('./train/Normal/Normal_' + str(i) + '.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 2
        t = t + 1
for i in range(213):
    T_Data = pd.read_csv('./train/PressureGain_constant/PressureGain_constant_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 3
        t = t + 1
for i in range(199):
    T_Data = pd.read_csv('./train/PropellerDamage_bad/PropellerDamage_bad_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 4
        t = t + 1
for i in range(208):
    T_Data = pd.read_csv('./train/PropellerDamage_slight/PropellerDamage_slight_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtrain[t] = Data_E1
        Ytrain[t] = 5
        t = t + 1
Xtrain1 = np.zeros([975, 180 * 16])
for i in range(975):
    for j in range(180):
        for k in range(16):
            Xtrain1[i][j * 16 + k] = Xtrain[i][j][k]

Xtest = np.zeros([245, 180, 16])
Ytest = np.zeros([245])
t = 0
for i in range(54):
    T_Data = pd.read_csv('./test/AddWeight/AddWeight_' + str(i) + '.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 1
        t = t + 1
for i in range(36):
    T_Data = pd.read_csv('./test/Normal/Normal_' + str(i) + '.csv', header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 2
        t = t + 1
for i in range(53):
    T_Data = pd.read_csv('./test/PressureGain_constant/PressureGain_constant_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 3
        t = t + 1
for i in range(50):
    T_Data = pd.read_csv('./test/PropellerDamage_bad/PropellerDamage_bad_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 4
        t = t + 1
for i in range(52):
    T_Data = pd.read_csv('./test/PropellerDamage_slight/PropellerDamage_slight_' + str(i) + '.csv',
                         header=None)  # Load CSV file
    T_Data = np.array(T_Data)  # Convert to numpy format
    if T_Data[:, 1].shape[0] >= 181:  # Determine if the collected data points exceed 180
        Data_E1 = T_Data[1:181, 1:]  # Intercept the first 180 values and discard the header
        Data_E1 = Normal(Data_E1)  # Standardize the signal
        Xtest[t] = Data_E1
        Ytest[t] = 5
        t = t + 1
Xtest1 = np.zeros([245, 180 * 16])
for i in range(245):
    for j in range(180):
        for k in range(16):
            Xtest1[i][j * 16 + k] = Xtest[i][j][k]

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



rfc = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc = rfc.fit(Xtrain1, Ytrain)
score_r = rfc.score(Xtest1, Ytest)
Ypred = rfc.predict(Xtest1)
print("Random Forest: {}".format(score_r))

Ytrain_n = np.zeros([5, len(Ytrain)])
for i in range(5):
    for j in range(len(Ytrain)):
        if Ytrain[j] == i + 1:
            Ytrain_n[i][j] = 1
Ytest_n = np.zeros([5, len(Ytest)])
for i in range(5):
    for j in range(len(Ytest)):
        if Ytest[j] == i + 1:
            Ytest_n[i][j] = 1
rfc1 = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc2 = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc3 = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc4 = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc5 = RandomForestClassifier(random_state=3407, n_estimators=1000)
rfc1 = rfc1.fit(Xtrain1, Ytrain_n[0])
print('1: ', rfc1.score(Xtest1, Ytest_n[0]))
rfc2 = rfc2.fit(Xtrain1, Ytrain_n[1])
print('2: ', rfc2.score(Xtest1, Ytest_n[1]))
rfc3 = rfc3.fit(Xtrain1, Ytrain_n[2])
print('3: ', rfc3.score(Xtest1, Ytest_n[2]))
rfc4 = rfc4.fit(Xtrain1, Ytrain_n[3])
print('4: ', rfc4.score(Xtest1, Ytest_n[3]))
rfc5 = rfc5.fit(Xtrain1, Ytrain_n[4])
print('5: ', rfc5.score(Xtest1, Ytest_n[4]))
Ypred1 = rfc1.predict(Xtest1)
Ypred2 = rfc2.predict(Xtest1)
Ypred3 = rfc3.predict(Xtest1)
Ypred4 = rfc4.predict(Xtest1)
Ypred5 = rfc5.predict(Xtest1)
Ypred_n = np.zeros([len(Ypred1)])
for i in range(len(Ypred1)):
    if Ypred1[i] == 1 and Ypred2[i] == 0 and Ypred3[i] == 0 and Ypred4[i] == 0 and Ypred5[i] == 0:
        Ypred_n[i] = 1
    elif Ypred3[i] == 1 and Ypred2[i] == 0 and Ypred1[i] == 0 and Ypred4[i] == 0 and Ypred5[i] == 0:
        Ypred_n[i] = 3
    else:
        Ypred_n[i] = Ypred[i]
print(classification_report(Ytest, Ypred_n))
print(accuracy_score(Ytest, Ypred_n))
# In[4]:

bag1 = BaggingClassifier()
bag2 = BaggingClassifier()
bag3 = BaggingClassifier()
bag4 = BaggingClassifier()
bag5 = BaggingClassifier()
bag1 = bag1.fit(Xtrain1, Ytrain_n[0])
print('1: ', bag1.score(Xtest1, Ytest_n[0]))
bag2 = bag2.fit(Xtrain1, Ytrain_n[1])
print('2: ', bag2.score(Xtest1, Ytest_n[1]))
bag3 = bag3.fit(Xtrain1, Ytrain_n[2])
print('3: ', bag3.score(Xtest1, Ytest_n[2]))
bag4 = bag4.fit(Xtrain1, Ytrain_n[3])
print('4: ', bag4.score(Xtest1, Ytest_n[3]))
bag5 = bag5.fit(Xtrain1, Ytrain_n[4])
print('5: ', bag5.score(Xtest1, Ytest_n[4]))
