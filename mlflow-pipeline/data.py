import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="load_data") as run:
    df = pd.read_csv("data/train.csv")

    df = df.drop(columns=['salad_bar'])
    
    df.to_csv("data/processed/data.csv", index=False)
    

