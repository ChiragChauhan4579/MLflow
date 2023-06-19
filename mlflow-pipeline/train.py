import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

with mlflow.start_run(run_name="train_model") as run:
    df = pd.read_csv("data/processed/data.csv")

    x = df.drop(columns=['cost'])
    y = df['cost']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    mlflow.log_metric("train_rmsle", mean_squared_log_error(y_train, model.predict(x_train)))
    mlflow.log_metric("test_rmsle", mean_squared_log_error(y_test, model.predict(x_test)))

    mlflow.sklearn.log_model(model, "rf-model")



