# path to raw data folder
RAW_PATH = 'data/raw/'

# path to processed data folder
PROCESSED_PATH = 'data/processed/'

# batched processed data
BATCH_FEATS_PATH = PROCESSED_PATH + 'feats_df_batches/'

# saved predictions
PREDS_PATH = 'data/predictions/'

# saved models
MODELS_PATH = 'models/'

# path to the model tb used in the flask app
APP_MODEL_PATH = MODELS_PATH + 'cat_1.cbm'

#length of windows and shifts
# SHIFTS = [1, 2, 3, 8, 12]
SHIFTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
WINS = [2, 5, 12]

#functions to be calculated as window aggregates
ROLL_FUNCS = ['sum', 'mean'] #, 'mean', 'std', 'min', 'max'

# min and max for months and IDs of shops and items. Extracted from respective raw files
COLS_MIN_MAX = {
    'shop_id':        (20, 40),#(26, 28), #(26, 28), (2, 59) (20, 40),
    'item_id':        (0, 22169), #(30, 22167)
    'date_block_num': (0, 33)
}

# columns groups for calculation of aggregate features
GROUP_COLS = {
    'shop':          ['shop_id', 'date_block_num'],
    'item':          ['item_id', 'date_block_num'],
    'shop_item':     ['shop_id', 'item_id', 'date_block_num'],
    'category':      ['item_category_id', 'date_block_num'],
    'shop_category': ['shop_id', 'item_category_id', 'date_block_num']
}

# Ho much zero target to include in train data, as percent of non-zero target
ZERO_PERC = .8

# batch size for features processing, number of unique shop_id in 1 batch
SHOPS_BATCH_SIZE = 8