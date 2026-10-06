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
        return self.pipe

    def predict(self, x_test):
        self.prediction = self.pipe.predict(x_test)
        return self.prediction
        
    def accuracy(self):
        return accuracy_score(self.test.authors, self.prediction) if not self.prediction else None
        
    def performance(self, target_names):
        if not self.prediction:
            return None
        self.report = classification_report(self.test.authors, self.prediction, target_names=target_names)
        return self.report

    def predict_category(self, text):
        """
        Predict the category of a given test using the trained classifier.
        """
        prediction = self.predict(text) #Important to use the in-class function to update the attribute as well
        return self.target_names[prediction[0]]
