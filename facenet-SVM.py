!pip install --upgrade tensorflow==2.0

import numpy as np
import cv2
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn import metrics
from tensorflow.keras.models import load_model

datasetpath = './'
num_person = len(os.listdir(datasetpath+'raw/'))
num_face = len(os.listdir(datasetpath+'raw/s1/'))
#print("There are",num_person,"subjects in this dataset, each has",num_face,"photos")
faces200 = np.load(datasetpath+'face200embedding.npy')

facenet = load_model(datasetpath+'facenet_keras.h5')
#facenet.summary()

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

addface = []
for i in [201,202,203,204]:
  for j in range(num_face):
    img = cv2.imread(datasetpath + "raw/s" + str(i) + '/' + str(j + 1) + '.jpg')
    reimg = cv2.resize(img, (160,160))
    addface.append(reimg)

addface = np.array(addface)
addface = prewhiten(addface)

addembedding = facenet.predict_on_batch(addface)
addembedding = l2_normalize(addembedding)
embs = np.concatenate((faces200,addembedding))

y = np.arange(1,num_person+1)
facelabel = np.repeat(y, num_face, axis=0)

X_train, X_test, y_train, y_test = train_test_split(embs, facelabel, random_state=0)

#print(X_train.shape, X_test.shape)

# this part is searching best SVM parameters
parameters = {'kernel': ['linear', 'rbf', "poly"], 'C': [0.1, 0.5, 1, 5.], 'gamma': ["scale", 0.001]}
metric = "f1_weighted"
svc = svm.SVC(random_state=0)
clf = GridSearchCV(svc, parameters, cv=5, scoring=metric)
clf.fit(X_train, y_train)
print(f"The best model uses {clf.best_params_} which results in a(n) {metric} score of {clf.best_score_}.")
# The best model uses {'C': 5.0, 'gamma': 'scale', 'kernel': 'linear'} which results in a(n) f1_weighted score of 0.8047896936552399.

y_pred = clf.predict(X_test)
#print(metrics.classification_report(y_test, y_pred))
print(metrics.f1_score(y_test,y_pred,average='weighted'))
print(metrics.precision_score(y_test,y_pred,average='weighted'))
print(metrics.recall_score(y_test,y_pred,average='weighted'))

model = clf.best_estimator_
joblib.dump(model, datasetpath+'models/FaceNet-SVM-model-full.joblib')

