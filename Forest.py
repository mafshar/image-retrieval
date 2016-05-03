import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


class RFC:

    def __init__(self, model=None):
        self.__model = model

    def train(self, features, labels, x_val, y_val):

        num_classifiers = [ 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350]
        max_val = 0
        for n in num_classifiers:
            rfc = RandomForestClassifier(n_estimators=n, criterion='entropy', n_jobs=-1, random_state=4321)
            rfc = rfc.fit(features, labels)
            prediction = rfc.predict(x_val)
            score = accuracy_score(y_val, prediction)
            print "n: " + str(n) + "\t" + "score: " + str(score)
            if score > max_val:
                max_val = score
                self.__model = rfc.fit(features, labels)

    def predict(self, key):
        result = self.__model.predict(key)
        return result


featuresL = np.load('For Miles/For Miles/train_cnn_code_last_layer.npy')
labels = np.load('For Miles/For Miles/train_y.npy')
x_valL = np.load('For Miles/For Miles/test_cnn_code_last_layer.npy')
y_val = np.load('For Miles/For Miles/real_test_y.npy')

print "Building Last Layer Random Forest Classifier..."
rfc = RFC()
print "Training RFC..."
train_time = time.time()
rfc.train(featuresL, labels, x_valL, y_val)
print "training time: " + str(time.time() - train_time)
print "Predicting..."
pred_time = time.time()
prediction = rfc.predict(x_valL)
print "Prediction time: " + str(time.time() - pred_time)
print "Score: " + str(accuracy_score(y_val, prediction))
