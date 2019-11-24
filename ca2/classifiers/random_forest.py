import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from ca2.ca2_params import *
from ca2.classifiers.classifier_wrapper import ClassifierWrapper


class RfClassifierWrapper(ClassifierWrapper):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def plot(self, save_fig=False):

        self.check_two_dimensions()
        X, y = self.get_data()

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolor='k')
        # plt.xlabel('Sepal length')
        # plt.ylabel('Sepal width')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-self.X_train.shape[0], azim=110)
        iris = datasets.load_iris()
        iris_shape = iris.data.shape
        X_shape = X.shape
        X_reduced = PCA(n_components=3).fit_transform(iris.data)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                   cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        if save_fig:
            plt.savefig("rf.png")
        plt.show()

    def plot2(self, save_fig=False):

        self.check_two_dimensions()
        X, y = self.get_data()

        # Parameters
        cmap = plt.cm.RdYlBu
        plot_step = 0.02  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        RANDOM_SEED = 13  # fix the seed on each iteration

        plot_idx = 1
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        scores = self.classifier.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(self.classifier)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(self.classifier, "estimators_"):
            model_details += " with {} estimators".format(
                len(self.classifier.estimators_))
        print(model_details + " with features",
              "has a score of", scores)

        # plt.subplot(3, 4, plot_idx)
        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers

        # Choose alpha blend level with respect to the number
        # of estimators
        # that are in use (noting that AdaBoost can use fewer estimators
        # than its maximum if it achieves a good enough fit early on)
        estimator_alpha = 1.0 / len(self.classifier.estimators_)
        for tree in self.classifier.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = self.classifier.predict(np.c_[xx_coarser.ravel(),
                                               yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

        plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
        plt.axis("tight")
        plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)

        if save_fig:
            plt.savefig('output_images\\rf.png')
        plt.show()
