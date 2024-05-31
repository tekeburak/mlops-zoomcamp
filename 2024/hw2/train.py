import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("homework-2")
    mlflow.sklearn.autolog()
    with mlflow.start_run():
                
        mlflow.set_tag("Developer", "Burak")
        mlflow.log_param("train-data-path", "./hw2_data/green_tripdata_2023-01.parquet")
        mlflow.log_param("val-data-path", "./hw2_data/green_tripdata_2023-02.parquet")
        mlflow.log_param("test-data-path", "./hw2_data/green_tripdata_2023-03.parquet")
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        max_depth, random_state = 10, 0
        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
