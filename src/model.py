from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class Classifier:
    def __init__(self, data, test, target_names):
        # Initialize attributes
        self.data = data
        self.test = test
        self.target_names = target_names
        
        # Initialize classifier
        self.clf = LinearSVC()
        
    def train(self):
        # Train the classifier
        self.clf.fit(self.data.x, self.data.y)
    
    def predict(self, x_test):
        self.prediction = self.clf.predict(x_test)
        return self.prediction
        
    def accuracy(self):
        return accuracy_score(self.test.y, self.prediction) if not self.prediction else None
        
    def performance(self, target_names):
        if not self.prediction:
            return None
        self.accuracy = accuracy_score(self.test.y, self.prediction)
        self.report = classification_report(self.test.y, self.prediction, target_names=target_names)
        return self.report

    def predict_category(self, text):
        """
        Predict the category of a given test using the trained classifier.
        """
        prediction = self.predict(text) #Important to use the in-class function to update the attribute as well
        return self.target_names[prediction[0]]
