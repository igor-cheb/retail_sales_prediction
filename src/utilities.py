import os
import glob
import numpy as np
import pandas as pd

from typing import Any
from itertools import product
from src.StackModel import StackModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from src.settings import COLS_MIN_MAX, SHIFTS, WINS, RAW_PATH, PROCESSED_PATH, BATCH_FEATS_PATH

def construct_cols_min_max(dfs: list[pd.DataFrame], 
                           cols: list[str]) -> dict:
    """Creates a dictionary of min max values of passed columns and passed datasets"""
    return {col: (min([el_df[col].min() for el_df in dfs]),
                  max([el_df[col].max() for el_df in dfs])) for col in cols}

def construct_fake_df(df_len: int=20000, 
                      lookback_window: int=15,
                      lags_list: list=[12, 8, 3, 2, 1]) -> tuple[pd.DataFrame, list]:
    """Function creates artificial sequence dataset. Used to experiment with RNN"""
    x = np.array(range(1, df_len))
    # y = np.sin(x/10) + x/100
    y = (np.sin(x/10)/x)*100 +  x/300

    X = []
    for ix, val in enumerate(y[:-(lookback_window-1)]):
        X.append(y[ix: ix+(lookback_window+1)][::-1])
    
    feats_cols = [f'lag_{col+1}' for col in range(lookback_window)]
    out_df = pd.DataFrame(X, columns=['target'] + feats_cols).dropna()
    
    if len(lags_list) > 0:
        filtered_cols  = [col for col in feats_cols if col in [f'lag_{num}' for num in lags_list]][::-1]
    else: 
        filtered_cols = feats_cols
    return out_df[['target'] + filtered_cols], filtered_cols

def check_folder(path: str, flash_folder: bool=True):
    """Function checks if path exists and creates folder if it does not"""
    if not os.path.exists(path):
        os.mkdir(path) 
    elif flash_folder:
        for filename in glob.glob(path + '/*'):
            os.remove(filename)

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
    """Subsamples rows with zero target to prevent the model from overfitting to 0"""
    local_df = df.copy()
    non_zero_df = local_df[local_df[target_col]!=0]
    zero_df = local_df[local_df[target_col]==0].sample(int(non_zero_df.shape[0] * zero_perc))
    return pd.concat([non_zero_df, zero_df], ignore_index=True)

def reduce_df_memory(df: pd.DataFrame):
    """Function casts down column types"""
    for col in df:
        if df[col].dtype == 'int64':
            df.loc[:, col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df.loc[:, col] = df[col].astype('float32')
    return df

def create_merged_raw(raw_folder_path: str=RAW_PATH, merged_name: str='merged_train_df'):
    """Function that merges and save raw files sales_train and items"""
    sales_train = pd.read_csv(raw_folder_path + 'sales_train.csv')
    items = pd.read_csv(raw_folder_path + 'items.csv')[['item_id', 'item_category_id']]
    merged = sales_train.merge(items, how='left', on='item_id')
    check_folder(PROCESSED_PATH)
    save_path = PROCESSED_PATH + f'{merged_name}.parquet'
    merged.to_parquet(save_path)
    
def read_train() -> pd.DataFrame:
    """Function reads and concats batched processed datasets with features"""
    all_dfs = []
    for path in glob.glob(BATCH_FEATS_PATH):
        all_dfs.append(pd.read_parquet(path))
    return pd.concat(all_dfs, ignore_index=True)


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
    # TODO: Consider adding validation dataset into data split and do early stopping
    
    all_months = np.array(sorted(df['date_block_num'].unique()))
    all_months = all_months[all_months >= max([max(SHIFTS), max(WINS)])]# leaving enough months for longest shift/window calculation
    cv_results = {'rmse':[], 'nrmse':[], 'train_months':[], 
                  'test_months':[], 'train_data':[], 'test_data':[], 
                  'pred':[], 'models':[]}
    
    for i, (train_index, test_index) in enumerate(months_cv_split.split(all_months)):
        train_months = all_months[train_index]
        test_months = all_months[test_index]
        
        train_df = df[df['date_block_num'].isin(train_months)]
        test_df = df[df['date_block_num'].isin(test_months)]
        cols_to_fit = ['date_block_num'] + cols_di['feats'] if type(model) == StackModel else cols_di['feats']
        model.fit(X=train_df[cols_to_fit], 
                  y=train_df[cols_di['target']].values.ravel())
        y_true = test_df[cols_di['target']].values
        y_pred = model.predict(test_df[cols_di['feats']]).round()
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
        cv_results['models'].append(model)
    if verbose in [1, 2]:
        print('\n' + '-'*30)
        print(f"RMSE mean: {np.mean(cv_results['rmse']):.2}")
        print(f"NRMSE mean: {np.mean(cv_results['nrmse']):.2}")
    return cv_results