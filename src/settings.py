# path to raw data folder
RAW_PATH = 'data/raw/'

# path to processed data folder
PROCESSED_PATH = 'data/processed/'

#length of windows and shifts
WINS_SHIFTS = [2, 6, 12]

#functions to be calculated as window aggregates
ROLL_FUNCS = ['sum', 'mean', 'std', 'min', 'max']


# SHOP_ID_MIN_MAX = (0, 59)
# ITEM_ID_MIN_MAX = (0, 22169)
# MONTH_MIN_MAX = (0, 33)

# min and max for months and IDs of shops and items. Extracted from respective raw files
COLS_MIN_MAX = {
    'shop_id': (0, 59), 
    'item_id': (0, 22169),
    'date_block_num': (0, 33)
}