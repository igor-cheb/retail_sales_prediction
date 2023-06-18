from itertools import product
import pandas as pd

from src.utilities import generate_backbone
from src.settings import PROCESSED_PATH, SHIFTS, WINS, ROLL_FUNCS, COLS_MIN_MAX, GROUP_COLS

class FeatureGenerator():
    """Class to generate all features used for training or inference"""

    def __init__(self):
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')
        # self.merged_df = self.merged_df[self.merged_df['shop_id'].isin([26, 27, 28])]

        self.index_cols = ['shop_id', 'item_id', 'date_block_num']
        self.base_cols = ['item_price', 'item_cnt_day']
        self.target_col = ['target']
        self.backbone = generate_backbone()
    
    def _gen_base_features(self) -> pd.DataFrame:
        """Adding shop_id level feature aggregates"""
        # local_df = self.merged_df[['date_block_num', 'shop_id', 'item_cnt_day', 'item_price']]\
        #                 .reset_index(drop=True).copy()
        # agg_di = {col: ROLL_FUNCS for col in base_cols}

        # local_df = local_df.groupby(self.shop_group_cols, as_index=False).agg(agg_di)\
        #                 .fillna(0)#.rename(columns={'item_id': 'deals'})
        # local_df.columns = ['_'.join(col) if col[1] else col[0] for col in local_df.columns ]
        
        # # local_df = local_df.reset_index()
        # local_df = self.shop_month_index_backbone.merge(local_df, 
        #                                                 how='left', 
        #                                                 on=self.shop_group_cols).fillna(0)
        # self.base_feat_cols = [f'{col}_{agg}' for col in base_cols for agg in ROLL_FUNCS]
        # return local_df

        local_df = self.merged_df[['date_block_num', 'item_id', 'shop_id', 
                                   'item_cnt_day', 'item_price']].reset_index(drop=True).copy()

        res_df = self.backbone.copy()
        agg_di = {col: ROLL_FUNCS for col in self.base_cols}

        for k, group in GROUP_COLS.items():
            agg_df = local_df.groupby(group, as_index = False).agg(agg_di)
            agg_df.columns = ['_'.join(col) + f'_per_{k}' if col[1] else col[0] for col in agg_df.columns ]
            res_df = res_df.merge(agg_df, how='left', on=group).fillna(0)

        res_df = res_df.rename(columns={'item_cnt_day_sum_per_shop_item': 'target'})
        self.base_feat_cols = [col for col in res_df if col not in 
                               self.index_cols + self.base_cols + self.target_col]
        return res_df

    def _add_shifts(self, 
                    df: pd.DataFrame, 
                    cols_to_shift: list[str]) -> pd.DataFrame:
        """Adding lag features to the passed df based on passed shift_cols columns"""
        # local_df = df.sort_values(self.shop_group_cols).copy()
        # self.lag_cols = []
        # for shift in SHIFTS + [0]:
        #     for col in shift_cols:
        #         if col in local_df:
        #             new_col = f'{col}_shift_{shift}'
        #             self.lag_cols.append(new_col)
        #             # this assignment works b/c it's same as joining by index and local_df is sorted by shop_group_cols
        #             local_df[new_col] = local_df.groupby('shop_id', as_index=False)[col].shift(periods=shift, fill_value=0)
        # return local_df
        # cols_to_shift = base_feat_cols + target_col
        self.shifted_cols = []
        for shift in SHIFTS:
            preshift_df = df[self.index_cols + cols_to_shift].copy()
            preshift_df['date_block_num'] = preshift_df['date_block_num'] + shift
            rename_dict = {col: f'{col}_lag_{shift}' for col in cols_to_shift}
            self.shifted_cols += [f'{col}_lag_{shift}' for col in cols_to_shift]
            preshift_df = preshift_df.rename(columns = rename_dict)
            
            df = df.merge(preshift_df, how='left', on=self.index_cols).fillna(0)
        return df

    def _add_rolling_windows(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """Adding window aggregates to the passed df based on passed cols_to_agg columns"""
        # local_df = df.sort_values(self.shop_group_cols)

        # self.roll_cols = []
        # for func in ROLL_FUNCS:
        #     for win_len in WINS:
        #         for col in cols_to_roll:
        #             if col in local_df:
        #                 new_name = f'{col}_roll_{func}_{win_len}'
        #                 self.roll_cols.append(new_name)
        #                 # this assignment works b/c it's same as joining by index and
        #                 # the dataset is ordered by shop and month
        #                 local_df[new_name] = local_df.groupby('shop_id').rolling(win_len, min_periods=1)\
        #                         .agg({col: func}).reset_index(drop=True).fillna(0)
        # return local_df
        local_df = df.copy()
        self.roll_cols = []
        col = self.target_col[0]
        group = GROUP_COLS['shop_item']
        sorted_df = df[group + [col]].drop_duplicates().sort_values(group).copy()
        for win_len in WINS:
            # groupping_k = col.split('_per_')[1] if not col == target_col[0] else 'shop_item'
            roll_df = sorted_df.copy()
            new_name = f'{col}_roll_mean_{win_len}'
            roll_df = roll_df.groupby(group[:-1], as_index=False)\
                             .rolling(win_len, on='date_block_num', 
                                      closed='right')[col].mean().fillna(0).reset_index()\
                             .rename(columns={col: new_name})
            local_df = local_df.merge(roll_df, how='left', on=group)
            self.roll_cols.append(new_name)
        return local_df
    
    def generate_features(self) -> pd.DataFrame:
        """Calculating all features and merging them in one dataset"""
        feats_df = self._gen_base_features()
        feats_df = self._add_shifts(df=feats_df, 
                                    cols_to_shift=(self.base_feat_cols + self.target_col))
        feats_df = self._add_rolling_windows(df=feats_df)
        return feats_df
