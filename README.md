# honest_home_task
WORK IN PROGRESS WORK IN PROGRESS WORK IN PROGRESS </br>
Implementation of a take home task as part of Honest bank recruitment process

Pipfile is provided to install all packages used.

Before running any notebook or flask app, please put the raw data into the `data/raw/` folder </br>
Raw data could be found here: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

To use the flask app:
<ul>
<li>Initiate the pipenv environment first: `pipenv shell`.</li>
<li>Start the flask endpoint that serves the trained model: `python src/endpoint.py`</li>
</ul>

Project structure:
<ul>
<li>data: raw and processed data used for training and inference</li>
<li>models: models saved during experiments</li>
<li>src: classes, settings and utilities used in modelling and data transformation</li>
<li>notebooks:</li>
    <ul>
    <li>EDA: short data exploration</li>
    <li>ETL: sandbox notebook for development and testing of feature engineering and cross validation </li>
    <li>modelling: sandbox for modelling experiments</li>
    <li>kaggle_submission: notebook to generate kaggle submissions</li>
    </ul>
</ul>

WORK IN PROGRESS WORK IN PROGRESS WORK IN PROGRESS
