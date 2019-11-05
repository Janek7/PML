from abc import ABC, abstractmethod


class ClassifierWrapper(ABC):
    classifier = None

    @abstractmethod
    def train(self, X_train, y_train):
        """
       trains a classifier with given training data
       :return:
       """
        pass

    @abstractmethod
    def validate(self, X_test, y_test):
        """
       validates a classifier with given test data and computes accuracy score
       :return: accuracy score
       """
        pass
