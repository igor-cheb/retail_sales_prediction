from itertools import product
import pandas as pd

from src.settings import PROCESSED_PATH, WINS_SHIFTS, ROLL_FUNCS

class FeatureGenerator():
    """Class to generate all features used for training or inference"""

    # TODO: shop_id_min etc. should be replaced with COLS_MIN_MAX from settings
    # TODO: it should be possible to merge _gen_deals_per_month_feats and _gen_revenue_per_shop_month_feats
    # TODO: _gen_shop_month_backbone tb moved to utilities and reused in TestGenerator as well
    
    def __init__(self):
        self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')
        self.shop_id_min, self.shop_id_max = self.merged_df['shop_id'].min(), self.merged_df['shop_id'].max()
        self.month_min, self.month_max = self.merged_df['date_block_num'].min(), self.merged_df['date_block_num'].max()
        
    def _gen_shop_month_backbone(self):
        """Creating dataframe where for each combination of shop and item every month is present"""
        self.shop_month_index_backbone = pd.DataFrame(product(
            range(self.shop_id_min, self.shop_id_max+1),
            range(self.month_min, self.month_max+1)
        ), columns = ['shop_id', 'date_block_num'])
    
    def _gen_deals_per_month_feats(self):
        """Calculating features based on number of deals per month"""
        # adding deals count column
        deals_cnt_df = self.merged_df[['date_block_num', 'shop_id', 'item_id']]\
                            .reset_index(drop=True).copy()
        group_cols = ['shop_id', 'date_block_num']
        deals_cnt_df = deals_cnt_df.sort_values(group_cols).groupby(group_cols)\
                            ['item_id'].count().reset_index()\
                            .rename(columns={'item_id': 'deals_cnt'})
        deals_cnt_df = self.shop_month_index_backbone.merge(deals_cnt_df, how='left', 
                                                            on=group_cols).fillna(0)

        # calculating lags
        deals_cnt_df = deals_cnt_df.set_index('shop_id')
        for shift in WINS_SHIFTS:
            deals_cnt_df[f'deals_cnt_shift_{shift}'] = deals_cnt_df.groupby('shop_id')['deals_cnt']\
                                                        .shift(periods=shift, fill_value=0)
        deals_cnt_df = deals_cnt_df.reset_index()

        # calculating rolling window aggregates
        deals_cnt_df = deals_cnt_df.sort_values(group_cols)
        for func in ROLL_FUNCS:
            for win_len in WINS_SHIFTS:
                deals_cnt_df[f'deals_cnt_roll_{func}_{win_len}'] = deals_cnt_df.groupby('shop_id').rolling(win_len, min_periods=1)\
                        .agg({'deals_cnt': func}).reset_index(drop=True).fillna(0)
        return deals_cnt_df
        

    def _gen_revenue_per_shop_month_feats(self):
        """Calculating features based on prices of sold items per month per shop"""
        # adding simple aggregates of prices over various deals
        prices_df = self.merged_df.reset_index()[['shop_id', 'date_block_num', 'item_price']]
        group_cols = ['shop_id', 'date_block_num']
        prices_df = prices_df.groupby(group_cols).agg({'item_price': ROLL_FUNCS}).fillna(0)
        prices_df.columns = ['_'.join(col) for col in prices_df.columns]
        prices_df = prices_df.reset_index()
        prices_df = self.shop_month_index_backbone.merge(prices_df, how='left').fillna(0)
        simple_agg_cols = [f'item_price_{agg}' for agg in ROLL_FUNCS]

        # adding lags
        prices_df = prices_df.sort_values(group_cols).set_index('shop_id')
        for shift in WINS_SHIFTS:
            for col in simple_agg_cols:
                prices_df[f'{col}_shift_{shift}'] = prices_df.groupby('shop_id')[col].shift(periods=shift, fill_value=0)
        prices_df = prices_df.reset_index()

        # adding window aggregates
        prices_df = prices_df.sort_values(group_cols)
        cols_to_agg = ['item_price_sum', 'item_price_mean']

        for func in ROLL_FUNCS:
            for win_len in WINS_SHIFTS:
                for col in cols_to_agg:
                    prices_df[f'{col}_roll_{func}_{win_len}'] = prices_df.groupby('shop_id').rolling(win_len, min_periods=1)\
                            .agg({col: func}).reset_index(drop=True).fillna(0)
        return prices_df
    
    def generate_features(self):
        """Calculating all features and merging them in one dataset"""
        self._gen_shop_month_backbone()
        deals_cnt_df = self._gen_deals_per_month_feats()
        prices_df = self._gen_revenue_per_shop_month_feats()
        return deals_cnt_df.merge(prices_df, how='left')