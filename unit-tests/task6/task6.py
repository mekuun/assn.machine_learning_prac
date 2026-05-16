import numpy as np
import typing


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.data_min) / (self.data_max - self.data_min)


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.data_mean) / self.data_std
