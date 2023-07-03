from sklearn.base import TransformerMixin, BaseEstimator

class DataReshaper(BaseEstimator, TransformerMixin):       
    def __init__(self, time_steps: int=12):
        self.time_steps = time_steps

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Method reshapes passed array so that 2nd dimension is broken down 
        into 2 dimensions: time steps and number of features
        """
        return X.reshape(-1, self.time_steps, int(X.shape[1]/self.time_steps))