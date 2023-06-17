from itertools import product
import pandas as pd

from src.utilities import generate_backbone
from src.settings import PROCESSED_PATH, WINS_SHIFTS, ROLL_FUNCS, COLS_MIN_MAX

class FeatureGenerator():
    """Class to generate all features used for training or inference"""

    # TODO: verify that new columns are created with correct indices

    def __init__(self):
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')
        self.shop_group_cols = ['shop_id', 'date_block_num']
        self.shop_month_index_backbone = generate_backbone(cols_for_backbone=self.shop_group_cols)
    
    def _gen_base_features(self):
        """Adding shop_id level feature aggregates"""
        local_df = self.merged_df[['date_block_num', 'shop_id', 'item_id', 'item_price']]\
                        .reset_index(drop=True).copy()
        agg_di ={
            'item_price': ROLL_FUNCS,
            'item_id': 'count'
        }
        local_df = local_df.groupby(self.shop_group_cols).agg(agg_di)\
                        .fillna(0).rename(columns={'item_id': 'deals'})
        local_df.columns = ['_'.join(col) for col in local_df.columns]
        local_df = local_df.reset_index()
        local_df = self.shop_month_index_backbone.merge(local_df, 
                                                        how='left', 
                                                        on=self.shop_group_cols).fillna(0)
        self.base_feat_cols = [f'item_price_{agg}' for agg in ROLL_FUNCS] + ['deals_count']
        return local_df

    def _add_shifts(self, 
                    df: pd.DataFrame, 
                    shift_cols: list[str]) -> pd.DataFrame:
        """Adding lag features to the passed df based on passed shift_cols columns"""
        local_df = df.sort_values(self.shop_group_cols).set_index('shop_id').copy()
        self.lag_cols = []
        for shift in WINS_SHIFTS:
            for col in shift_cols:
                new_col = f'{col}_shift_{shift}'
                self.lag_cols.append(new_col)
                local_df[new_col] = local_df.groupby('shop_id')[col].shift(periods=shift, fill_value=0)
        return local_df.reset_index()

    def _add_rolling_windows(self,
                             df: pd.DataFrame,
                             cols_to_agg: list[str]) -> pd.DataFrame:
        """Adding window aggregates to the passed df based on passed cols_to_agg columns"""
        local_df = df.sort_values(self.shop_group_cols)

        self.roll_cols = []
        for func in ROLL_FUNCS:
            for win_len in WINS_SHIFTS:
                for col in cols_to_agg:
                    new_name = f'{col}_roll_{func}_{win_len}'
                    self.roll_cols.append(new_name)
                    local_df[new_name] = local_df.groupby('shop_id').rolling(win_len, min_periods=1)\
                            .agg({col: func}).reset_index(drop=True).fillna(0)
        return local_df
    
    def generate_features(self):
        """Calculating all features and merging them in one dataset"""
        feats_df = self._gen_base_features()
        feats_df = self._add_shifts(df=feats_df, shift_cols=self.base_feat_cols)
        cols_to_roll = ['item_price_sum', 'item_price_mean', 'deals_count']
        feats_df = self._add_rolling_windows(df=feats_df, cols_to_agg=cols_to_roll)
        return feats_df
