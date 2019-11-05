import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from params import *


def load_data():
    """
    load the data, extract features and labels and split into training and test data
    :return:
    """
    # read all lines, extract header and filter lines
    lines = [line.split(csv_separator) for idx, line in
             enumerate(open(input_data_file, 'r').read().split(csv_line_separator))
             if line != '']
    feature_col_indices = [idx for idx, feature in enumerate(lines[0][:-1]) if feature in feature_cols]
    lines = [line for idx, line in enumerate(lines[1:]) if eval(line_pre_filter)]

    # label set
    label_indices_set = list(set([line[-1] for line in lines]))

    # filter features and append calc_features
    data = []
    for line in lines:
        new_line = [float(e) for idx, e in enumerate(line) if idx in feature_col_indices]
        for calc_feature in calc_features:
            new_line.append(calc_feature(new_line))
        data.append(new_line)

    # transform to numpy arrays
    X = np.array([np.array(line) for line in data])
    y = np.array([label_indices_set.index(line[-1]) for line in lines])

    # split and return
    return train_test_split(X, y, test_size=1 - trainig_size)


def plot(classifier, X, y):
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
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
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


def knn():
    """
    trains a knn classifier and show / return results
    :return: classifier and accuracy score
    """
    X_train, X_test, y_train, y_test = load_data()
    # train classifier
    classifier = neighbors.KNeighborsClassifier(k, weights=weights, algorithm=algorithm)
    classifier.fit(X_train, y_train)

    # plot if two dimensions
    if X_train[0].shape[0] == 2:
        plot(classifier, X_train, y_train)

    # test classifier
    predictions = classifier.predict(X_test)
    return classifier, accuracy_score(y_test, predictions)


if __name__ == '__main__':

    classifier, accuracy_score = knn()
    print('Accuracy score: {}'.format(accuracy_score))
