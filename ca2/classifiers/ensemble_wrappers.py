from ca2.ca2_params import *
from ca2.classifiers.classifier_wrapper import ClassifierWrapper
from sklearn.ensemble import VotingClassifier
# from mlxtend.classifier import StackingClassifier


class VotingClassifierWrapper(ClassifierWrapper):

    def __init__(self, estimators):
        self.classifier = VotingClassifier(estimators, voting=voting)

    def plot(self, save_fig=False):
        pass


class StackingClassifierWrapper(ClassifierWrapper):

    def __init__(self, estimators, meta_estimator):
        # self.classifier = StackingClassifier(classifiers=estimators, meta_classifier=meta_estimator,
        #                                      use_probas=use_probas)
        pass

    def plot(self, save_fig=False):
        pass
