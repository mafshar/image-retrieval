from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


if __name__ == "__main__":
    # Modify this to get feature, y
    feature = []
    y = []
    ovr_svm = GridSearchCV(OneVsRestClassifier(SVC(), cv=10, param_grid{"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, n_jobs=-1)
    ovo_svm = GridSearchCV(SVC(decision_function_shape='ovo'), cv=10, param_grid{"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, n_jobs=-1)
    ovr_sum.fit(feature, y)
    ovo_svm.fit(feature, y)
