import pandas as pd
from itertools import product

from src.settings import COLS_MIN_MAX

def generate_backbone(cols_for_backbone: list[str]=['shop_id', 'item_id', 'date_block_num']
                        ) -> pd.DataFrame:
    # creating dataframe where for each combination of shop and item every month is present
    ranges = [range(COLS_MIN_MAX[col][0], COLS_MIN_MAX[col][1]+1) for col in cols_for_backbone]
    index_backbone = pd.DataFrame(product(*ranges), columns = cols_for_backbone)
    return index_backbone