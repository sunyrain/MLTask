import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_set = np.load('trainset.npy')
test_set = np.load('testset.npy')

possible_status = ['0: AddWeight', '1: Normal', '2: PressureGain_constant', '3: PropellerDamage_bad', '4: PropellerDamage_slight']

X_train = train_set[:, 0:16]
y_train = train_set[:, 16]
X_test = test_set[:, 0:16]
y_test = test_set[:, 16]

rfc = RandomForestClassifier(random_state=3407, n_estimators=1000)

rfc = rfc.fit(X_train, y_train)

score_r = rfc.score(X_test, y_test)

print("Random Forest: {}" .format(score_r))

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.legend(possible_status)
plt.show()