# honest_home_task
WORK IN PROGRESS WORK IN PROGRESS WORK IN PROGRESS  


  
Implementation of a take home task as part of Honest bank recruitment process

Before running any notebook or flask app, please put the raw data into the `data/raw/` folder </br>
Raw data could be found here: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

To use the flask app:

- Put raw data into the `data/raw/` folder
- Initiate the pipenv environment and install necessary packages:

      pipenv shell
      pipenv install

- Start the flask endpoint that serves the trained model:

      python src/endpoint.py

Project structure:

- data: raw and processed data used for training and inference
- models: models saved during experiments
- src: classes, settings and utilities used in modelling and data transformation
- notebooks:
  - EDA: short data exploration
  - ETL: sandbox notebook for development and testing of feature engineering and cross validation
  - modelling: sandbox for modelling experiments
  - kaggle_submission: notebook to generate kaggle submissions

    
WORK IN PROGRESS WORK IN PROGRESS WORK IN PROGRESS
