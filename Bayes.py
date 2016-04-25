from sklearn.naive_bayes import GaussianNB as nb


class Bayes:

    def __init__(self, model=None):
        self.__model = model

    def train(self, x, label):
        self.__model = nb().fit(x, label)

    def predict(self, key):
        result = self.__model.predict(key)
        return result


if __name__ == "__main__":

    feature = []
    label = []
    model = Bayes().train(feature, label)