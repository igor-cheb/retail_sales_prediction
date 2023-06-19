# import pandas as pd
# from itertools import product

# from src.FeatureGenerator import *
# from src.utilities import generate_backbone

# class TestGenerator():
#     """
#     Class to generate test datasets for crossvalidation or 
#     generate features for a provided ready shop_id-item_id index backbone.
#     """
#     # TODO: consider building a separate TargetGenerator class with 
#     # generate_all_targets and generate_target_for_month
#     # then pass it along with FeatureGenerator and merge it here

#     def __init__(self):
#         pass
#     def add_month_to_backbone(self, 
#                               shop_item_backbone: pd.DataFrame, 
#                               month_num: int) -> pd.DataFrame:
#         """
#         Function simply takes in a backbone file and adds a column of month to it.
#         Used to merge feature columns afterwards.
#         We are adding month-1 b/c passed month_num is the number of month we are predicting for.
#         This separate function is needed to allow to pass a realy backbone file like test.csv.
#         """
#         shop_item_backbone['date_block_num'] = month_num - 1
#         return shop_item_backbone

#     def add_features_to_backbone(self, 
#                                  test_backbone: pd.DataFrame,
#                                  feat_generator: FeatureGenerator) -> pd.DataFrame:
#         """
#         Function takes in test backbone, i.e. df of shops and items and a given month, 
#         calls feature generator and returns the merged result.
#         """
#         feats = feat_generator.generate_features()
#         return test_backbone.merge(feats, how='left')
    








#     # def generate_shop_item_backbone(self, test_size: int) -> pd.DataFrame:
#     #     """
#     #     Function generates backbone dataframe with 
#     #     shop_id, item_id of the passed size.
#     #     """
#     #     # TODO control for percentage of item_ids and shop_ids not seen in training, keep under 10%
#     #     backbone = generate_backbone(cols_for_backbone=['shop_id', 'item_id'])
#     #     return backbone.sample(test_size).reset_index(drop=True)