from sklearn.svm import LinearSVC


def classify(train_features, train_labels, test_features):
    # linear SVM model over these features
    return LinearSVC().fit(train_features, train_labels).predict(test_features)
