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


train_set = np.load('train.npy')
test_set = np.load('test.npy')


possible_status = ['0: AddWeight', '1: Normal', '2: PressureGain_constant', '3: PropellerDamage_bad', '4: PropellerDamage_slight']

X_train = train_set[:, 0:-1]
y_train = train_set[:, -1]
X_test = test_set[:, 0:-1]
y_test = test_set[:, -1]


rfc = RandomForestClassifier(random_state=3407)

#X_train, X_test = apply_feature_extraction(X_train, X_test, y_train)


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
plt.legend(possible_status)
plt.show()