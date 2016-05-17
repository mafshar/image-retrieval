import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import time
import pickle


class RFC:

    def __init__(self, model=None):
        self.__model = model

    def train(self, features, labels, x_val, y_val):

        num_classifiers = [340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350]
        max_val = 0
        for n in num_classifiers:
            rfc = RandomForestClassifier(n_estimators=n, criterion='entropy', n_jobs=-1, random_state=4321)
            rfc = rfc.fit(features, labels)
            prediction = rfc.predict(x_val)
            score = accuracy_score(y_val, prediction)
            print "n: " + str(n) + "\t" + "error: " + str(1-score)
            if score > max_val:
                max_val = score
                self.__model = rfc.fit(features, labels)

    def predict(self, key):
        result = self.__model.predict(key)
        return result

    def get_estimator(self, idx):

        decision_tree_estimators = self.__model.estimators_
        return decision_tree_estimators[idx]



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
print "Error: " + str(1-accuracy_score(y_val, prediction))

print "Visualizing..."

clf = rfc.get_estimator(100)

original_names = pickle.load(open('train_dict.p')).values()

dog_cat_map = ["dog", "cat", "cat", "dog", "dog", "cat", "dog", "dog", "dog", "cat", "dog",
       "dog", "cat", "cat", "cat", "dog", "cat", "dog", "dog",
       "dog", "dog", "dog", "dog", "dog", "dog", "dog", "cat"]


dot_data1 = StringIO()
tree.export_graphviz(clf, out_file=dot_data1, max_depth=2, class_names=original_names)
graph1 = pydot.graph_from_dot_data(dot_data1.getvalue())
graph1.write_pdf("breeds.pdf")

dot_data2 = StringIO()
tree.export_graphviz(clf, out_file=dot_data2, max_depth=2, class_names=dog_cat_map)
graph2 = pydot.graph_from_dot_data(dot_data2.getvalue())
graph2.write_pdf("dogvcat.pdf")