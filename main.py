import cv2
import numpy as np
import joblib
from trial import test_model
from tensorflow.keras.models import load_model

points1 = []
def mouse_pos(event,x,y,flags,params):
    global points1
    if event == cv2.EVENT_LBUTTONDOWN:
        points1 = [x,y]
        return points1


def main():
    Letmein = cv2.imread('letMeIn.png')
    switch = cv2.imread('Switch.png')
    facenet = load_model('models/facenet_keras.h5')
    clf = joblib.load('models/FaceNet-SVM-model-full.joblib')

    classifier = "SVM Classifier"
    print("Default Classifier : SVM Classifier")

    global points1


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    count = 0
    frame_no = 1
    result = ""
    while 1:
        is_ok, img1 = cap.read()
        frame_no = frame_no + 1
        if not is_ok:
            break
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        switch = cv2.resize(switch, (200, 65))
        img = np.copy(img1)
        img[img.shape[0] - switch.shape[0] - 50:img.shape[0] - 50, img.shape[1] - switch.shape[1] - 50:img.shape[1] - 50] = switch

        Letmein = cv2.resize(Letmein, (200, 65))
        img[img.shape[0] - Letmein.shape[0] - 50:img.shape[0] - 50, 50:Letmein.shape[1] + 50] = Letmein

        if len(points1) == 2:
            if points1[0] < 251 and points1[0] > 51 and points1[1] < 430 and points1[1] > 367:
                print("Recogonizing Face using " + classifier)
                points1 = [0, 0]
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                print(faces)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        crop_img = img1[y-50:y + w + 50, x-50:x + h + 50]
                        cv2.imshow('cropped', crop_img)
                        cv2.waitKey()

                        result = test_model(crop_img,facenet,clf)
                else:
                    print("Failed to detect face. Try again!")
            elif points1[0] < 591 and points1[0] > 390 and points1[1] < 430 and points1[1] > 366:
                if classifier == "MLP Classifier":
                    classifier = "SVM Classifier"
                elif classifier == "SVM Classifier":
                    classifier = "MLP Classifier"
                print("Switching to " + classifier + " classifier")
                points1 = [0, 0]
            else:
                result = ""

        if result == "Access Denied":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(img, result, (int(img.shape[0] / 4) - 25, int(img.shape[1] / 2) - 90), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
        cv2.imshow('img', img)
        cv2.namedWindow("img")
        cv2.setMouseCallback('img', mouse_pos)
        if result == "Access Denied":
            key = cv2.waitKey(5000)
            result = ""

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
