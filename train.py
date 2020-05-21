import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import joblib

def train_model(datasetpath):
    faces = np.load(datasetpath+'facedata2_gray.npy')
    num_person = 200
    num_face = 14
    y = np.arange(1,num_person+1)
    facelabel = np.repeat(y, num_face, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(faces, facelabel, random_state=0)

    pca = PCA(n_components=150, whiten=True)
    pca.fit(X_train)
    joblib.dump(pca, 'models/PCA.joblib')

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    parameters = {'kernel': ('linear', 'rbf', "poly"), 'C': [
            0.1, 0.5, 1, 5.], 'gamma': ["scale", 0.001]}

    metric = "f1_weighted"

    svc = svm.SVC(random_state=0)
    clf = GridSearchCV(svc, parameters, cv=5, scoring=metric)
    clf.fit(X_train_pca, y_train)

    print(f"The best model uses {clf.best_params_} which results in a(n) {metric} score of {clf.best_score_}.")

    model = clf.best_estimator_

    joblib.dump(model, 'models/PCA-SVM-model.joblib')

    y_pred = clf.predict(X_test_pca)
    print(y_pred)
    return clf, pca


if __name__ == '__main__':
    datasetpath = './'
    train_model(datasetpath)
