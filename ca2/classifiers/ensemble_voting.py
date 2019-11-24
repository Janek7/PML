from ca2.ca2_params import *
from ca2.classifiers.classifier_wrapper import ClassifierWrapper
from sklearn.ensemble import VotingClassifier


class VotingClassifierWrapper(ClassifierWrapper):

    def __init__(self, estimators):
        self.classifier = VotingClassifier(estimators, voting=voting)

    def plot(self, save_fig=False):
        pass
