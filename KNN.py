import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import model_selection


class KNN:

    def __init__(self, model=None):
        self.__model = model

    def train(self, x, y):

        classifier = knn()
        parameter = {'n_neighbors':[3, 4, 5], 'algorithm':['auto', 'ball_tree', 'kd_tree']}
        clf = model_selection.GridSearchCV(classifier, parameter)
        self.__model = clf

    def predict(self, key):
        result = self.__model.predict(key)
        return result

if __name__ == "__main__":

    feature = []
    label = []
    model = KNN()
    model.train(feature, label)