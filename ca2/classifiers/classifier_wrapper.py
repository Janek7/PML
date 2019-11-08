from abc import abstractmethod, ABC

import numpy as np
from sklearn.metrics import accuracy_score


class ClassifierWrapper(ABC):
    classifier = None
    X_train, y_train, X_test, y_test = None, None, None, None

    def train(self, X_train, y_train):
        """
       trains a classifier with given training data
       :return:
       """
        self.X_train = X_train
        self.y_train = y_train
        self.classifier.fit(X_train, y_train)

    def validate(self, X_test, y_test):
        """
       validates a classifier with given test data and computes accuracy score
       :return: accuracy score
       """
        self.X_test = X_test
        self.y_test = y_test
        predictions = self.classifier.predict(X_test)
        return accuracy_score(y_test, predictions)

    def check_two_dimensions(self):
        """
        checks if the data set has two dimensions. If not throw an exception
        :param X: data set as numpy array
        :return:
        """
        if self.get_data()[0][0].shape[0] != 2:
            raise Exception('Only two dimensional data sets can be plotted')

    @abstractmethod
    def plot(self, save_fig=False):
        """
        plots the trained classifier with training data
        :param save_fig: image should be saved?
        :return:
        """
        pass

    def get_data(self):
        """
        concatenates training and test data
        :return: tupel of X and y
        """
        return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))
