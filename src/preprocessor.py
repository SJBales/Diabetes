import pandas as pd

# --------- Reading Data ---------
features = pd.read_parquet("data/diabetes_features.parquet")
target = pd.read_parquet("data/diabetes_target.parquet")

