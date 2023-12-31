import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.decomposition import PCA


def apply_feature_extraction(Xtrain, Xtest):
    model = PCA(n_components=0.99)
    print(model.n_components)
    Xtrain_transformed = model.fit_transform(Xtrain)
    Xtest_transformed = model.transform(Xtest)
    print(f"Reduced number of features: {Xtrain_transformed.shape[1]}")
    return Xtrain_transformed, Xtest_transformed


X_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

estimators = np.linspace(300, 1000, 8)
accuracy = []
for n in range(len(estimators)):
    rfc = RandomForestClassifier(random_state=0, n_estimators=int(estimators[n]), criterion='gini')
    rfc = rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    score_r = rfc.score(X_test, y_test)
    accuracy.append(score_r)
    cm = confusion_matrix(y_test, y_pred)
    print("Random Forest Accuracy: {}".format(score_r))

plt.figure()
plt.plot(estimators, accuracy)
plt.grid()
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

#print(classification_report(y_test, y_pred))
#print(cm)

plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
#plt.legend(possible_status)
#plt.show()