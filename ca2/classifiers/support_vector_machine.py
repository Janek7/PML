import numpy as np
from sklearn import svm
import pylab as pl
from sklearn.decomposition import PCA

from ca2.ca2_params import *
from ca2.classifiers.classifier_wrapper import ClassifierWrapper


class SvmClassifierWrapper(ClassifierWrapper):

    def __init__(self):
        self.classifier = svm.SVC(kernel=kernel, gamma=gamma, C=C)

    def plot(self, save_fig=False):

        self.check_two_dimensions()
        X, y = self.get_data()

        pca = PCA(n_components=2).fit(X)
        pca_2d = pca.transform(X)
        svmClassifier_2d = svm.LinearSVC(random_state=111).fit(pca_2d, y)
        for i in range(0, pca_2d.shape[0]):
            if y[i] == 0:
                c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', s=50, marker='+')
            elif y[i] == 1:
                c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', s=50, marker='o')
            elif y[i] == 2:
                c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', s=50, marker='*')
        pl.legend([c1, c2, c3], ['Setosa', 'Versicolor', 'Virginica'])
        x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
        y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
        Z = svmClassifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pl.contour(xx, yy, Z)
        pl.title('Support Vector Machine Decision Surface')
        pl.axis('off')

        if save_fig:
            pl.savefig('output_images\\svm.png')
        pl.show()
