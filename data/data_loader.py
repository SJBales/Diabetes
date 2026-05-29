from sklearn.datasets import fetch_openml
import pandas as pd


def get_data():
    # Reading the data
    diabetes = fetch_openml("Diabetes130US", version=1, as_frame=True)

    # Parsing the features and target
    features = pd.DataFrame(diabetes['data'])
    target = pd.DataFrame(diabetes['target'])

    # Writing the features adn target to parquets
    features.to_parquet("diabetes_features.parquet")
    target.to_parquet("diabetes_target.parquet")


if __name__ == "__main__":
    get_data()
