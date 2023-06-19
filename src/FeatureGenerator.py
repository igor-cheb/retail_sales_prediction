from typing import  Optional
import pandas as pd

from src.utilities import generate_backbone
from src.settings import PROCESSED_PATH, SHIFTS, WINS, ROLL_FUNCS, COLS_MIN_MAX, GROUP_COLS

class FeatureGenerator():
    """Class to generate all features used for training or inference"""

    # TODO: join categories column
    
    def __init__(self, target_months: Optional[list]=None):
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')

        self.index_cols: list[str] = ['shop_id', 'item_id', 'date_block_num']
        self.base_cols: list[str] = ['item_price', 'item_cnt_day']
        self.target_col: list[str] = ['target']

        self.target_months = target_months
        # if test data is generated for particular months, we want to make sure
        # that those months are in the min max range of COLS_MIN_MAX
        if target_months:
            new_max_month = max(target_months)
            new_min_month = min([new_max_month, COLS_MIN_MAX['date_block_num'][0]])
            COLS_MIN_MAX['date_block_num'] = (new_min_month, new_max_month)
        self.backbone = generate_backbone(cols_min_max=COLS_MIN_MAX)
    
    def _gen_base_features(self) -> pd.DataFrame:
        """Adding shop_id level feature aggregates"""
        local_df = self.merged_df[['date_block_num', 'item_id', 'shop_id', 
                                   'item_cnt_day', 'item_price']].reset_index(drop=True).copy()

        res_df = self.backbone.copy()
        agg_di = {col: ROLL_FUNCS for col in self.base_cols}

        for k, group in GROUP_COLS.items():
            agg_df = local_df.groupby(group, as_index = False).agg(agg_di)
            agg_df.columns = ['_'.join(col) + f'_per_{k}' if col[1] else col[0] for col in agg_df.columns ]
            res_df = res_df.merge(agg_df, how='left', on=group).fillna(0)

        res_df = res_df.rename(columns={'item_cnt_day_sum_per_shop_item': 'target'})
        self.base_feat_cols = [str(col) for col in res_df if col not in 
                                          self.index_cols + self.base_cols + self.target_col]
        return res_df

    def _add_shifts(self, 
                    df: pd.DataFrame, 
                    cols_to_shift: list[str]) -> pd.DataFrame:
        """Adding lag features to the passed df based on passed shift_cols columns"""
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
        local_df = df.copy()
        self.roll_cols = []
        col = self.target_col[0]
        group = GROUP_COLS['shop_item']
        sorted_df = df[group + [col]].drop_duplicates().sort_values(group).copy()
        for win_len in WINS:
            roll_df = sorted_df.copy()
            new_name = f'{col}_roll_mean_{win_len}'
            # closed param makes sure current row is not included in the window
            roll_df = roll_df.groupby(group[:-1], as_index=False)\
                             .rolling(win_len, on='date_block_num', 
                                      closed='left')[col].mean().fillna(0).reset_index()\
                             .rename(columns={col: new_name})
            local_df = local_df.merge(roll_df, how='left', on=group)
            self.roll_cols.append(new_name)
        return local_df
    
    def generate_features(self) -> pd.DataFrame:
        """Calculating all features and merging them in one dataset"""
        feats_df = self._gen_base_features()
        feats_df = self._add_shifts(df=feats_df, 
                                    cols_to_shift=self.base_feat_cols + self.target_col)
        feats_df = self._add_rolling_windows(df=feats_df)

        out_cols = self.index_cols + self.target_col + self.shifted_cols + self.roll_cols
        if self.target_months:
            return feats_df[feats_df['date_block_num'].isin(self.target_months)][out_cols]
        else: 
            return feats_df[out_cols]
    
    def add_features_to_backbone(self,
                                 test_backbone: pd.DataFrame) -> pd.DataFrame:
        """
        Function takes in test backbone, i.e. df of shops and items, 
        calls feature generator and returns the merged result.
        """
        if self.target_months:
            test_backbone['date_block_num'] = self.target_months[0] 
        feats = self.generate_features()
        return test_backbone.merge(feats, how='left').fillna(0)
