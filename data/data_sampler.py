import pandas as pd

features = pd.read_parquet("data/diabetes_features.parquet")

# Sampling 1% of the data for mock batch inference
sampled_features = features.sample(frac=0.01)

sampled_features.to_parquet("data/inference_feature_example.parquet")
