# path to raw data folder
RAW_PATH = 'data/raw/'

# path to processed data folder
PROCESSED_PATH = 'data/processed/'

#length of windows and shifts
# SHIFTS = [1, 2, 6, 12]
SHIFTS = [1, 2, 3, 8, 12]
# SHIFTS = [2, 4, 8, 12]
WINS = [2, 5, 12]

#functions to be calculated as window aggregates
ROLL_FUNCS = ['sum', 'mean'] #, 'mean', 'std', 'min', 'max'

# min and max for months and IDs of shops and items. Extracted from respective raw files
COLS_MIN_MAX = {
    'shop_id': (26, 28),#(0, 59), 
    'item_id': (0, 22169),
    'date_block_num': (0, 33)
}

# columns groups for aggregate calculation
GROUP_COLS = {
    'shop': ['shop_id', 'date_block_num'],
    'item': ['item_id', 'date_block_num'],
    'shop_item': ['shop_id', 'item_id', 'date_block_num']
}

# Ho much zero target to include in train data, as percent of non-zero target
ZERO_PERC = .8