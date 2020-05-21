from flask import Flask, render_template,request
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import cv2
import os
import glob
import re
import base64
import time
from sklearn import svm
import io
from sklearn import metrics
from trial import test_model
from tensorflow.keras.models import load_model
import joblib

classifier_name = 2
facenet = load_model('models/facenet_keras.h5')
clf = joblib.load('models/FaceNet-SVM-model-full.joblib')
clf1 = load_model('models/FaceNet-NN-model.h5')
facenet._make_predict_function()
clf1._make_predict_function()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index2.html',name=name)

@app.route('/clf',methods=["POST"])
def classvalue():
    global classifier_name
    classifier_name = request.form['classifierName']
    return classifier_name

@app.route('/photo',methods=["POST"])
def disp_pic():
    global classifier_name, clf, clf1, facenet, face_cascade


    img_data = request.form['imgData']
    try:
        classifier_name = classvalue()
    except:
        pass
    encoded_data = img_data.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    count = 0
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            crop_img = img[y - 50:y + w + 50, x - 50:x + h + 50]
            if classifier_name == 1:
                result, color = test_model(crop_img, facenet, clf,classifier_name)
            else:
                result, color = test_model(crop_img, facenet, clf1,classifier_name)
            if result == "Access Granted":
                count = count + 1
        if count > 0:
            cv2.putText(img, "Access Granted", (int(img.shape[0] / 4) - 25, int(img.shape[1] / 2) - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, result, (int(img.shape[0] / 4) - 25, int(img.shape[1] / 2) - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "RETRY!!!", (int(img.shape[0] / 4) - 25, int(img.shape[1] / 2) - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(3000)
    cv2.destroyWindow('image')
    return img_data

def main():
    app.run()
    app.debug = True

if __name__ == '__main__':
    main()