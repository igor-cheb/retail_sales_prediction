import pandas as pd
import numpy as np

class ToyModel():
    """Baseline model that predicts mean of the passed y vector at train time"""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.y_mean = y.mean()

    def predict(self, X):
        return np.array([self.y_mean] * X.shape[0])