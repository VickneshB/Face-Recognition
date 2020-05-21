!pip install --upgrade tensorflow==2.0

import numpy as np
import tensorflow as tf
import cv2
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input

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

y_train = tf.keras.utils.to_categorical(y_train-1)
y_test = tf.keras.utils.to_categorical(y_test-1)

inputs = Input(shape=(128,))
x = Flatten()(inputs)
x = Dense(512,'relu')(x)
predictions = Dense(num_person,'softmax')(x)

face_complete = Model(inputs, predictions)
face_complete.summary()

face_complete.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
face_complete.fit(X_train, y_train, epochs=50, batch_size=128)

test_scores = face_complete.evaluate(X_test, y_test)
acc = test_scores[1]
print(acc)

face_complete.save(datasetpath+'models/FaceNet-NN-model.h5')

