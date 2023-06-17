import pandas as pd
import numpy as np
from typing import Any
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.settings import COLS_MIN_MAX, WINS_SHIFTS

def generate_backbone(cols_for_backbone: list[str]=['shop_id', 'item_id', 'date_block_num']
                     ) -> pd.DataFrame:
    """Creating dataframe with all combinations of passed columns values are present"""
    ranges = [range(COLS_MIN_MAX[col][0], COLS_MIN_MAX[col][1]+1) for col in cols_for_backbone]
    index_backbone = pd.DataFrame(product(*ranges), columns = cols_for_backbone)
    return index_backbone

def run_cv(df: pd.DataFrame, 
           months_cv_split: TimeSeriesSplit, 
           model: Any,
           cols_di: dict,
           verbose: bool=False) -> dict:
    """
    Function that performs cross validation using passed model 
    over the passed df with all features and month column and passed
    splitter by months
    """
    all_months = df['date_block_num'].unique()
    all_months = all_months[all_months > max(WINS_SHIFTS)] # leaving enough months for longest shift/window calculation

    cv_results = {'rmse':[], 'nrmse':[], 'train_months':[], 'test_months':[]}

    for i, (train_index, test_index) in enumerate(months_cv_split.split(all_months)):
        train_months = all_months[train_index]
        test_months = all_months[test_index]
        
        train_df = df[df['date_block_num'].isin(train_months)]
        test_df = df[df['date_block_num'].isin(test_months)]

        model.fit(X=train_df[cols_di['feats']], y=train_df[cols_di['target']])
        y_true = test_df[cols_di['target']].values
        y_pred = model.predict(X=test_df[cols_di['feats']])
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred)**(.5)
        nrmse = rmse / np.std(y_true) # (np.percentile(y_true, 75) - np.percentile(y_true, 25))

        if verbose:
            print(f"Fold {i}:")
            print(f"  Train months: {all_months[train_index]}")
            print(f"  Test months: {all_months[test_index]}")
            print(f'  NRMSE: {nrmse: .2}')
            print(f'  RMSE : {rmse: .2}\n')

        cv_results['rmse'].append(rmse); cv_results['nrmse'].append(nrmse)
        cv_results['train_months'].append(all_months[train_index])
        cv_results['test_months'].append(all_months[test_index])
    if verbose:
        print('\n' + '-'*30)
        print(f"RMSE mean: {np.mean(cv_results['rmse']):.2}")
        print(f"NRMSE mean: {np.mean(cv_results['nrmse']):.2}")
    return cv_results