from typing import  Optional
import pandas as pd
import gc

from src.utilities import generate_backbone, balance_zero_target, construct_cols_min_max, \
    reduce_df_memory, check_folder
from src.settings import PROCESSED_PATH, RAW_PATH, SHIFTS, WINS, ROLL_FUNCS, \
    COLS_MIN_MAX, GROUP_COLS, ZERO_PERC, SHOPS_BATCH_SIZE, BATCH_FEATS_PATH

class FeatureGenerator():
    """Class to generate all features used for training or inference."""

    #TODO: consider adding difference between lags and/or rolls as features
    
    def __init__(self, verbose: bool=False):
        self.verbose = verbose

        # reading file with raw data
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')
        
        # creating lists of default columns
        self.index_cols = ['shop_id', 'item_id', 'date_block_num']
        self.base_cols =  ['item_price', 'item_cnt_day']
        self.target_col = ['target']
        self.cat_col =    ['item_category_id']

        # creating item-category mapping
        item_cat_df = pd.read_csv(RAW_PATH + 'items.csv')
        self.item_cat_map = item_cat_df[['item_id', 'item_category_id']].drop_duplicates()

        # creating or cleaning folder for processed data
        check_folder(BATCH_FEATS_PATH)

    def _gen_base_features(self, backbone) -> pd.DataFrame:
        """Adding shop_id level feature aggregates"""
        local_df = self.merged_df[['date_block_num', 'item_id', 
                                   'shop_id', 'item_cnt_day', 
                                   'item_price', 'item_category_id']].reset_index(drop=True).copy()

        res_df = backbone.copy()
        agg_di = {col: ROLL_FUNCS for col in self.base_cols}

        for k, group in GROUP_COLS.items():
            agg_df = local_df.groupby(group, as_index = False).agg(agg_di)
            agg_df.columns = ['_'.join(col) + f'_per_{k}' if col[1] else col[0] for col in agg_df.columns ]
            res_df = res_df.merge(agg_df, how='left', on=group).fillna(0)

        res_df = res_df.rename(columns={'item_cnt_day_sum_per_shop_item': 'target'})
        self.base_feat_cols = [str(col) for col in res_df if col not in 
                               self.index_cols + self.base_cols + 
                               self.target_col + self.cat_col]
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

    def _add_rolling_windows(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def generate_features(self, 
                          cols_min_max: dict=COLS_MIN_MAX,
                          balance_target_by_zero: bool=True) -> pd.DataFrame:
        """Calculating all features and merging them in one dataset"""
        backbone = generate_backbone(cols_min_max=cols_min_max).merge(self.item_cat_map, how='left')
        feats_df = self._gen_base_features(backbone=backbone)
        feats_df = reduce_df_memory(feats_df)
        del backbone
        if self.verbose: print('base feats done')
        shops = feats_df['shop_id'].unique()
        all_shops_num = len(shops) // SHOPS_BATCH_SIZE; 
        if (len(shops) % SHOPS_BATCH_SIZE) > 0: all_shops_num += 1
        cnt = 0
        if self.verbose: print(f'{all_shops_num} batches')

        all_out_dfs = []
        for ix in range(0, len(shops), SHOPS_BATCH_SIZE):
            cnt+=1
            cur_shops = shops[ix:ix + SHOPS_BATCH_SIZE]
            loc_feats_df = feats_df[feats_df['shop_id'].isin(cur_shops)].copy()
            loc_feats_df = self._add_shifts(df=loc_feats_df, 
                                            cols_to_shift=self.base_feat_cols + self.target_col)
            if self.verbose: print('shifts done')
            loc_feats_df = self._add_rolling_windows(df=loc_feats_df)
            if self.verbose: print('rolls done')
            out_cols = self.index_cols + self.cat_col + self.shifted_cols + self.roll_cols + self.target_col
            # out_df = loc_feats_df[out_cols]
            loc_feats_df = reduce_df_memory(loc_feats_df)
            batch_df = loc_feats_df[out_cols] if not balance_target_by_zero else balance_zero_target(df=loc_feats_df[out_cols], 
                                                                                                     zero_perc=ZERO_PERC, 
                                                                                                     target_col=self.target_col[0])

            all_out_dfs.append(batch_df)
            batch_df.to_parquet(BATCH_FEATS_PATH + f'feats_df_batch_{cnt}.parquet')
            if self.verbose:
                print(f'batch {cnt}/{all_shops_num} done')
                print('-'*30)
        if self.verbose: print('concatenating')
        gc.collect()
        out_df = pd.concat(all_out_dfs, ignore_index=True)
        return out_df 

    def add_features_to_backbone(self,
                                 test_backbone: pd.DataFrame,
                                 target_month: int) -> pd.DataFrame:
        """
        Function takes in test backbone, i.e. df of shops and items, 
        calls feature generator and returns the merged result.
        Used to generate features for data/raw/test.csv.
        """
        test_backbone['date_block_num'] = target_month
        custom_cols_min_max = construct_cols_min_max(dfs=[test_backbone],
                                                     cols=self.index_cols)
        custom_cols_min_max['date_block_num'] = (custom_cols_min_max['date_block_num'][1] - max(WINS), 
                                                 custom_cols_min_max['date_block_num'][1])
        feats_df = self.generate_features(cols_min_max=custom_cols_min_max, 
                                          balance_target_by_zero=False)
        return test_backbone.merge(feats_df.drop(self.target_col[0], axis=1), how='left').fillna(0)
