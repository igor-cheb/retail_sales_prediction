# Prediction of sales for a retail chain
Implementation of a take home task as part of recruitment process for a banking startup

Before running any notebook or flask app, please put the raw data into the `data/raw/` folder </br>
Raw data could be found here: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

To use the flask app:

- Put raw data into the `data/raw/` folder
- Initiate the pipenv environment and install necessary packages:

      pipenv shell
      pipenv install

- Start the flask endpoint that serves the trained model:

      python src/endpoint.py
  URL query example:
  
      http://localhost:8000/?shop_id=5&shop_id=1&shop_id=10&item_id=11&item_id=8&item_id=44
Project structure:

- data: raw and processed data used for training and inference
- models: models saved during experiments
- src: classes, settings and utilities used in modelling and data transformation
- notebooks:
  - EDA: short data exploration
  - ETL: sandbox notebook for development and testing of feature engineering and cross validation
  - modelling: sandbox for modelling experiments
  - kaggle_submission: notebook to generate kaggle submissions
<br><br><br>
  
