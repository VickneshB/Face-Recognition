import cv2
import numpy as np
#import joblib
#import glob
#from tensorflow.keras.models import load_model

def test_model(x_test,facenet,clf,classifier_name):
#def test_model(x_test):
    #facenet = load_model('models/facenet_keras.h5')
    #clf = joblib.load('models/FaceNet-SVM-model.joblib')

    x_test = cv2.resize(x_test, (160, 160))
    x_test = np.expand_dims(prewhiten(x_test),axis=0)
    #cv2.imshow('',x_test[0,:,:,:])
    new_x_test = l2_normalize(facenet.predict(x_test))
    if classifier_name == 1:
        y_predicted = clf.predict(new_x_test)
    else:
        y_predicted = np.argmax(clf.predict(new_x_test)) + 1
    print("prediction for ",y_predicted)
    if y_predicted > 200:
        return "Access Granted", (0, 255, 0)
    else:
        return "Access Denied", (0, 0, 255)
    
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

if __name__ == '__main__':
    datasetpath = './'
    test_model(datasetpath)
