import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from ca2.ca1_params import *
from ca2.classifiers.classifier_wrapper import ClassifierWrapper


class KnnClassifierWrapper(ClassifierWrapper):

    def train(self, X_train, y_train):
        # train classifier
        self.classifier = neighbors.KNeighborsClassifier(k, weights=weights, algorithm=algorithm)
        self.classifier.fit(X_train, y_train)

        # plot if two dimensions
        if X_train[0].shape[0] == 2:
            self.plot(X_train, y_train)

    def validate(self, X_test, y_test):
        predictions = self.classifier.predict(X_test)
        return accuracy_score(y_test, predictions)

    def plot(self, X, y):
        """
        plots the given data with classes
        :param X: data
        :param y: labels
        :return:
        """
        h = .02  # step size in the mesh
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (k, weights))

        # plt.savefig("knn.png")
        plt.show()


if __name__ == '__main__':

    classifier = KnnClassifierWrapper()
    # classifier.train()
    # accuracy_score = classifier.validate()
    # print('Accuracy score: {}'.format(accuracy_score))
