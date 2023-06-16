import pandas as pd
from itertools import product

from src.FeatureGenerator import *
from src.settings import COLS_MIN_MAX #SHOP_ID_MIN_MAX, ITEM_ID_MIN_MAX

class TestGenerator():
    """
    Class to generate test datasets for crossvalidation or 
    generate features for a provided ready shop_id-item_id index backbone.
    """
    # TODO: consider building a separate TargetGenerator class with 
    # generate_all_targets and generate_target_for_month
    # then pass it along with FeatureGenerator and merge it here

    def __init__(self, train: bool=False):
        # TODO: merged_df is read in FeatureGenerator too, should be optimised
        if train:
            self.merged_df = pd.read_parquet(PROCESSED_PATH + 'merged_train_df.parquet')

    def _generate_backbone(self, 
                          cols_for_backbone: list[str]=['shop_id', 'item_id', 'date_block_num']
                          ) -> pd.DataFrame:
        # creating dataframe where for each combination of shop and item every month is present
        ranges = [range(COLS_MIN_MAX[col][0], COLS_MIN_MAX[col][1]+1) for col in cols_for_backbone]
        index_backbone = pd.DataFrame(product(*ranges), columns = cols_for_backbone)
        return index_backbone

    # 2 functions below are used for training only
    
    def generate_all_targets(self):
        """
        Function creates 1 month lookahead target for all possible combinations of 
        shops, items and months. Used for training only.
        """
        # TODO: consider saving result of this function and reading it in init if train=True is passed

        # creating groupping for particular month, shop and item
        grouping_cols = ['shop_id', 'item_id', 'date_block_num']
        target_df = self.merged_df[grouping_cols + ['item_cnt_day']].sort_values(grouping_cols)
        target_df = target_df.groupby(grouping_cols)['item_cnt_day'].sum().reset_index()\
                        .rename(columns={'item_cnt_day':'sum_sales_cnt'})

        # generating backbone with all combinations of the index columns
        index_backbone = self._generate_backbone()

        # merging aggregated initial df with the backbone to calculate target correctly
        extended_target_df = index_backbone.merge(target_df, how='left', on=grouping_cols).fillna(0)
        extended_target_df = extended_target_df.sort_values(grouping_cols)

        # grouping by shop_id and item_id and shifting by 1 row "into the future"
        extended_target_df['target'] = extended_target_df.groupby(grouping_cols[:-1])['sum_sales_cnt'].shift(-1)
        return extended_target_df

    def generate_target_for_month(self, month_nums: list[int]):
        """Function to generate target column for particular month. Used for training only."""
        # generating all possible targets
        all_target = self.generate_all_targets()
        
        # because we are predicting for next month, 
        # we are picking only rows with current month equal to target month-1
        months_before_target = [el-1 for el in month_nums]
        all_target = all_target[all_target['date_block_num'].isin(months_before_target)]

        # including rows with zero and non-zero target 
        # to equal extent not to overfit the model to either
        all_target = pd.concat(
            [all_target[all_target['target']>0], 
            all_target[all_target['target']==0].sample(all_target[all_target['target']>0].shape[0])],
            ignore_index=True
        )
        
        return all_target

    # 3 functions below are used for inference only

    def generate_shop_item_backbone(self, test_size: int) -> pd.DataFrame:
        """
        Function generates backbone dataframe with 
        shop_id, item_id approximately of the passed size.
        TB used for cross validation.
        """
        # TODO control for percentage of item_ids and shop_ids not seen in training, keep under 10%
        backbone = self._generate_backbone(cols_for_backbone=['shop_id', 'item_id'])
        return backbone.sample(test_size).reset_index(drop=True)

    def add_month_to_backbone(self, 
                              shop_item_backbone: pd.DataFrame, 
                              month_num: int) -> pd.DataFrame:
        """
        Function simply takes in a backbone file and adds a column of month to it.
        Used to merge feature columns afterwards.
        We are adding month-1 b/c passed month_num is the number of month we are predicting for.
        This separate function is needed to allow to pass a realy backbone file like test.csv.
        """
        shop_item_backbone['date_block_num'] = month_num - 1
        return shop_item_backbone

    def add_features_to_backbone(self, 
                                 test_backbone: pd.DataFrame,
                                 feat_generator: FeatureGenerator) -> pd.DataFrame:
        """
        Function takes in test backbone, i.e. df of shops and items and a given month, 
        calls feature generator and returns the merged result.
        """
        feats = feat_generator.generate_features()
        return test_backbone.merge(feats, how='left')