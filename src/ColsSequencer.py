from sklearn.base import TransformerMixin, BaseEstimator

class ColsSequencer(BaseEstimator, TransformerMixin):       
    def __init__(self, time_steps: int=12):
        self.time_steps = time_steps

    def fit(self, X, y=None):
        """
        Function creates ordered set of columns, a sequence of identical gorups of cols
        where each group corresponds to a next time step
        """
        num_char_len = len(f'_{self.time_steps}') # exploiting the columns names format here
        base_columns = [col[:-num_char_len] for col in X.columns if f'_{self.time_steps}' in col]
        self.ordered_col_names = []
        for i in range(self.time_steps, 0, -1):
            self.ordered_col_names += [f'{col}_{i}' for col in base_columns]
        return self
    
    def transform(self, X):
        """Simply reorders the columns of the passed df according to the col list from self.fit"""
        return X[self.ordered_col_names]