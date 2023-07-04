import sys 
import numpy as np
import pandas as pd

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from catboost import CatBoostRegressor
from flask import Flask, request, jsonify

from src.FeatureGenerator import FeatureGenerator
from src.settings import APP_MODEL_PATH

app = Flask(__name__)

@app.route('/')
def model_app():
    # getting shops and items
    shop_ids = request.args.getlist("shop_id", type=float)
    item_ids = request.args.getlist("item_id", type=float)
    
    # generating features
    test_backbone = pd.DataFrame(np.column_stack([shop_ids, item_ids]), 
                                 columns = ['shop_id', 'item_id']).astype('int')
    feat_generator = FeatureGenerator(verbose=False, save_files=False)
    features_df = feat_generator.add_features_to_backbone(test_backbone=test_backbone, target_month=34)

    # predicting
    cat_from_file = CatBoostRegressor()
    cat_model = cat_from_file.load_model(APP_MODEL_PATH)
    pred = cat_model.predict(data=features_df[cat_model.feature_names_])

    return jsonify(pred.tolist())

if __name__ == '__main__':
    app.run(debug=True, port=8000)