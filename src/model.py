from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Classifier:
    def __init__(self):
        self.pipe = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC())])
        self.prediction = None

    def train(self, x_train, y_train):
        self.pipe.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        self.prediction = self.pipe.predict(x_test)
        return self.prediction

    def accuracy(self, y_true):
        return accuracy_score(y_true, self.prediction)

    def performance(self, y_true):
        return classification_report(y_true, self.prediction)

    def predict_category(self, features):
        return self.predict(features)[0]
