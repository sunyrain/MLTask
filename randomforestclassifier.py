import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from sklearn.decomposition import PCA


def apply_feature_extraction(Xtrain, Xtest):
    model = PCA(n_components=0.99)
    print(model.n_components)
    Xtrain_transformed = model.fit_transform(Xtrain)
    Xtest_transformed = model.transform(Xtest)
    print(f"Reduced number of features: {Xtrain_transformed.shape[1]}")
    return Xtrain_transformed, Xtest_transformed


X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


#X_train, X_test = apply_feature_extraction(X_train, X_test)

#possible_status = ['0: AddWeight', '1: Normal', '2: PressureGain_constant',
# '3: PropellerDamage_bad', '4: PropellerDamage_slight']

#X_train, X_test = apply_feature_extraction(X_train, X_test, y_train)

rfc = RandomForestClassifier(random_state=0, n_estimators=1000)         # 3407
rfc = rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

score_r = rfc.score(X_test, y_test)
print("Random Forest: {}" .format(score_r))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
#plt.legend(possible_status)
#plt.show()