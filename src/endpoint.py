

from flask import Flask, request
import pandas as pd
import numpy as np

import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.FeatureGenerator import FeatureGenerator

app = Flask(__name__)

@app.route('/')
def model_app():
    shop_ids = request.args.getlist("shop_id", type=float)
    item_ids = request.args.getlist("item_id", type=float)
    
    test_backbone = pd.DataFrame(np.column_stack([shop_ids, item_ids]), 
                                 columns = ['shop_id', 'item_id']).astype('int')
    print(test_backbone)
    feat_generator = FeatureGenerator(verbose=False, save_files=False)
    feats = feat_generator.add_features_to_backbone(test_backbone=test_backbone, target_month=34)
    
    return feats.values.tolist()

if __name__ == '__main__':
    app.run(debug=True, port=8000)