import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from scipy import ndimage
import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        centered_x = np.array([self.center_potential(matrix) for matrix in x])
        return centered_x.reshape((centered_x.shape[0], -1))

    def center_potential(self, x):
        mask = (x < 20).astype(float)
        sx, sy = x.shape[0]//2, x.shape[1]//2
        (cx, cy) = ndimage.center_of_mass(mask)
        shiftx, shifty = int(sx - cx), int(sy - cy)
        x = np.roll(np.roll(x, shiftx, axis=0), shifty, axis=1)
        return x


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([
        ('transformer', PotentialTransformer()),
        ('pca', TruncatedSVD(n_components=10)),
        ('decision_tree', ExtraTreesRegressor(n_estimators=900, criterion="friedman_mse", max_depth=20, n_jobs=-1))
    ])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)

    return {file: value for file, value in zip(test_files, predictions)}
