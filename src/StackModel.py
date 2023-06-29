from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

from src.settings import SHIFTS, WINS

class StackModel():
    """Class to stack models in several layers for time series data"""
    def __init__(self, 
                 lvl_1_models: list, 
                 lvl_2_models: list,
                 month_col: str,
                 lvl_1_feats: list,
                 target_col: str,
                 train_ratio: float=.6, # data volume percentage used by 1st level models over 2nd lvl model
                 verbose: bool=False):
        self.lvl_1_models = lvl_1_models
        self.lvl_2_models = lvl_2_models
        self.month_col = month_col
        self.train_ratio = train_ratio
        self.lvl_1_feats = lvl_1_feats
        self.target_col = target_col
        self.verbose = verbose

    def _fit_all_models(self, models: list, 
                        df: pd.DataFrame, 
                        label_col: pd.Series) -> list:
        """Function fits all passed models to the passed data and returnes trained models"""
        fitted_models = []
        for i, model in enumerate(models):
            model.fit(df, label_col)
            fitted_models.append(model)
            if self.verbose: print(f'model {i} training done')
        return fitted_models

    def _train_1_lvl(self, models: list, 
                    all_months: np.ndarray, 
                    splitter: TimeSeriesSplit,
                    df: pd.DataFrame, 
                    label_col: pd.Series,
                    label_col_name: str,
                    months_col: str) -> tuple[list, pd.DataFrame]:
        """Function trains base models of a stacked arch"""
        
        # generating predictions for next lvl models through rolling window CV
        all_pred = [] # next_months = []; 
        train_cols = [col for col in df if col!=months_col]
        for train_index, test_index in splitter.split(all_months):
            if self.verbose: 
                print(all_months[train_index])
                print(all_months[test_index])

            X_train = df[df[months_col].isin(all_months[train_index])][train_cols]
            X_test = df[df[months_col].isin(all_months[test_index])][train_cols]
            y_train = label_col[df[months_col].isin(all_months[train_index])]
            y_test = label_col[df[months_col].isin(all_months[test_index])]

            pred = np.array([all_months[test_index]] * X_test.shape[0])
            for i, model in enumerate(models):
                model.fit(X_train, y_train)
                if self.verbose: print(f'model {i} training done')
                pred = np.column_stack([pred, model.predict(X_test)])
            pred = np.column_stack([pred, y_test])
            all_pred.append(pred)
        feats_cols = [f'model_{k}' for k in range(len(models))]
        out_columns = [months_col] + feats_cols + [label_col_name]

        # fitting models to all available data
        fitted_models = self._fit_all_models(models=models, 
                                             df=df[train_cols],
                                             label_col=label_col)
        return fitted_models, pd.DataFrame(np.vstack(all_pred), columns=out_columns)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Method to fit the stacked model to the passed data"""
        all_months = np.array(sorted(X[self.month_col].unique()))
        all_months = all_months[all_months >= max([max(SHIFTS), max(WINS)])]# leaving enough months for longest shift/window calculation
        train_size_1 = int(len(all_months) * self.train_ratio)
        
        tscv_1 = TimeSeriesSplit(test_size = 1, 
                                 max_train_size=train_size_1, 
                                 n_splits=len(all_months) - train_size_1)
        
        self.fitted_models_1, pred_for_lvl_2 = self._train_1_lvl(models=self.lvl_1_models,
                                                                 all_months=all_months,
                                                                 splitter=tscv_1,
                                                                 df=X,
                                                                 label_col=y,
                                                                 label_col_name=self.target_col,
                                                                 months_col=self.month_col
                                                                 )
        
        train_cols_lvl_2 = [col for col in pred_for_lvl_2 if col not in [self.month_col, self.target_col]]
        self.fitted_models_2 = self._fit_all_models(models=self.lvl_2_models, 
                                                    df=pred_for_lvl_2[train_cols_lvl_2],
                                                    label_col=pred_for_lvl_2[self.target_col]
                                                    )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Method to predict using stacked model"""
        pred = []; cols_lvl_2 = []
        for i, model in enumerate(self.fitted_models_1):
            pred.append(model.predict(X[self.lvl_1_feats]))
            cols_lvl_2.append(f'model_{i}')
        lvl_1_pred = pd.DataFrame(np.column_stack(pred), columns=cols_lvl_2)
        pred = []

        if self.fitted_models_2:
            for model in self.fitted_models_2:
                pred.append(model.predict(lvl_1_pred))
            lvl_2_pred = np.column_stack(pred).mean(axis=1)
        else:
            lvl_2_pred = lvl_1_pred.mean(axis=1)
        return lvl_2_pred