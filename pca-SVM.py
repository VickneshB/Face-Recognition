import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics

datasetpath = './face_dataset_1/'

num_person = len(os.listdir(datasetpath+'raw/'))
num_face = len(os.listdir(datasetpath+'raw/s1/'))
#print("There are",num_person,"subjects in this dataset, each has",num_face,"photos")
faces = np.load(datasetpath+'facedata2_gray.npy')

fig = plt.figure(figsize=(20, 10))
# plot several images
for i in range(28):
    ax = fig.add_subplot(4, 7, i + 1, xticks=[], yticks=[])
    ax.imshow(faces[i].reshape(160,160),cmap=plt.cm.gray)

y = np.arange(1,num_person+1)
facelabel = np.repeat(y, num_face, axis=0)

X_train, X_test, y_train, y_test = train_test_split(faces, facelabel, random_state=0)

#print(X_train.shape, X_test.shape)

pca = PCA(n_components=150, whiten=True)
pca.fit(X_train)

#print(pca.explained_variance_ratio_[0:20])

#plt.imshow(pca.mean_.reshape(160,160),cmap=plt.cm.gray)

#print(pca.components_.shape)

# visulize principal components
fig = plt.figure(figsize=(25, 15))
for i in range(0,100,5):
  ax = fig.add_subplot(4,5,i/5+1,xticks=[],yticks=[])
  ax.imshow(pca.components_[i].reshape(160,160),cmap=plt.cm.gray)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print(X_train_pca.shape)
#print(X_test_pca.shape)

clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

fig = plt.figure(figsize=(8, 5))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(160,160),cmap=plt.cm.gray)
    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(facelabel[y_pred],
                 fontsize='small', color=color)

y_pred = clf.predict(X_test_pca)
#print(metrics.classification_report(y_test, y_pred))
print(metrics.f1_score(y_test,y_pred,average='weighted'))
print(metrics.precision_score(y_test,y_pred,average='weighted'))
print(metrics.recall_score(y_test,y_pred,average='weighted'))


