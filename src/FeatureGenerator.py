from itertools import product
import pandas as pd

from src.settings import PROCESSED_PATH, WINS_SHIFTS, ROLL_FUNCS, COLS_MIN_MAX

class FeatureGenerator():
    """Class to generate all features used for training or inference"""

    # TODO: shop_id_min etc. should be replaced with COLS_MIN_MAX from settings
    # TODO: _gen_shop_month_backbone tb moved to utilities and reused in TestGenerator as well

    def __init__(self):
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')
        self.shop_id_min, self.shop_id_max = COLS_MIN_MAX['shop_id'][0], COLS_MIN_MAX['shop_id'][1]
        self.month_min, self.month_max = COLS_MIN_MAX['date_block_num'][0], COLS_MIN_MAX['date_block_num'][1]
        
        self.shop_group_cols = ['shop_id', 'date_block_num']

    def _gen_shop_month_backbone(self):
        """Creating dataframe where for each combination of shop and item every month is present"""
        self.shop_month_index_backbone = pd.DataFrame(product(
            range(self.shop_id_min, self.shop_id_max+1),
            range(self.month_min, self.month_max+1)
        ), columns = ['shop_id', 'date_block_num'])
    
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

        return local_df

    def _add_shifts(self, 
                    df: pd.DataFrame, 
                    shift_cols: list[str]) -> pd.DataFrame:
        """Adding lag features to the passed df based on passed shift_cols columns"""
        local_df = df.sort_values(self.shop_group_cols).set_index('shop_id').copy()
        for shift in WINS_SHIFTS:
            for col in shift_cols:
                local_df[f'{col}_shift_{shift}'] = local_df.groupby('shop_id')[col].shift(periods=shift, fill_value=0)
        return local_df.reset_index()

    def _add_rolling_windows(self,
                             df: pd.DataFrame,
                             cols_to_agg: list[str]) -> pd.DataFrame:
        """Adding window aggregates to the passed df based on passed cols_to_agg columns"""
        local_df = df.sort_values(self.shop_group_cols)

        for func in ROLL_FUNCS:
            for win_len in WINS_SHIFTS:
                for col in cols_to_agg:
                    local_df[f'{col}_roll_{func}_{win_len}'] = local_df.groupby('shop_id').rolling(win_len, min_periods=1)\
                            .agg({col: func}).reset_index(drop=True).fillna(0)
        return local_df
    
    def generate_features(self):
        """Calculating all features and merging them in one dataset"""
        self._gen_shop_month_backbone()
        feats_df = self._gen_base_features()
        simple_agg_cols = [f'item_price_{agg}' for agg in ROLL_FUNCS] + ['deals_count']
        feats_df = self._add_shifts(df=feats_df, shift_cols=simple_agg_cols)
        roll_cols = ['item_price_sum', 'item_price_mean', 'deals_count']
        feats_df = self._add_rolling_windows(df=feats_df, cols_to_agg=roll_cols)
        return feats_df
