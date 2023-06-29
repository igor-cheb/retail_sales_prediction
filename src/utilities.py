import pandas as pd
import numpy as np
from typing import Any
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.StackModel import StackModel
from src.settings import COLS_MIN_MAX, SHIFTS, WINS

def construct_cols_min_max(dfs: list[pd.DataFrame], 
                           cols: list[str]) -> dict:
    """Creates a dictionary of min max values of passed columns and passed datasets"""
    return {col: (min([el_df[col].min() for el_df in dfs]),
                  max([el_df[col].max() for el_df in dfs])) for col in cols}

def generate_backbone(cols_for_backbone: list[str]=['shop_id', 'item_id', 'date_block_num'],
                      cols_min_max: dict=COLS_MIN_MAX
                     ) -> pd.DataFrame:
    """Creating dataframe with all combinations of passed columns values are present"""
    ranges = [range(cols_min_max[col][0], cols_min_max[col][1]+1) for col in cols_for_backbone]
    index_backbone = pd.DataFrame(product(*ranges), columns = cols_for_backbone)
    return index_backbone

def balance_zero_target(df: pd.DataFrame, 
                        zero_perc: float,  # between 0 an 1, what percent of zero target to output
                        target_col: str 
                        ) -> pd.DataFrame:
    """Subsamples rows with zero target to prevent the model to overfitting to 0"""
    local_df = df.copy()
    non_zero_df = local_df[local_df[target_col]!=0]
    zero_df = local_df[local_df[target_col]==0].sample(int(non_zero_df.shape[0] * zero_perc))
    return pd.concat([non_zero_df, zero_df], ignore_index=True)

def run_cv(df: pd.DataFrame, 
           months_cv_split: TimeSeriesSplit, 
           model: Any,
           cols_di: dict,
           verbose: int=0) -> dict:
    """
    Function that performs cross validation using passed model 
    over the passed df with all features and month column and passed
    splitter by months
    """
    # TODO: Control for the percentage of new customers and shops in the test data not seen in training
    
    all_months = np.array(sorted(df['date_block_num'].unique()))
    all_months = all_months[all_months >= max([max(SHIFTS), max(WINS)])]# leaving enough months for longest shift/window calculation
    cv_results = {'rmse':[], 'nrmse':[], 'train_months':[], 
                  'test_months':[], 'train_data':[], 'test_data':[], 'pred':[]}
    
    for i, (train_index, test_index) in enumerate(months_cv_split.split(all_months)):
        train_months = all_months[train_index]
        test_months = all_months[test_index]
        
        train_df = df[df['date_block_num'].isin(train_months)]
        test_df = df[df['date_block_num'].isin(test_months)]
        cols_to_fit = ['date_block_num'] + cols_di['feats'] if type(model) == StackModel else cols_di['feats']
        model.fit(X=train_df[cols_to_fit], 
                  y=train_df[cols_di['target']].values.ravel())
        y_true = test_df[cols_di['target']].values
        y_pred = model.predict(test_df[cols_di['feats']])
        
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred)**(.5)
        nrmse = rmse / np.std(y_true) # (np.percentile(y_true, 75) - np.percentile(y_true, 25))

        if verbose == 2:
            print(f"Fold {i}:")
            print(f"  Train months: {all_months[train_index]}, size: {len(train_df):,}")
            print(f"  Test months: {all_months[test_index]},   size: {len(test_df):,}")
            print(f'  NRMSE: {nrmse: .2}')
            print(f'  RMSE : {rmse: .2}\n')

        cv_results['rmse'].append(rmse); cv_results['nrmse'].append(nrmse)
        cv_results['train_months'].append(all_months[train_index])
        cv_results['test_months'].append(all_months[test_index])
        cv_results['train_data'].append(train_df)
        cv_results['test_data'].append(test_df)
        cv_results['pred'].append(y_pred)
    if verbose in [1, 2]:
        print('\n' + '-'*30)
        print(f"RMSE mean: {np.mean(cv_results['rmse']):.2}")
        print(f"NRMSE mean: {np.mean(cv_results['nrmse']):.2}")
    return cv_results