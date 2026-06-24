import joblib
from preprocessor import inferencePreprocessor
import pandas as pd

# Loading the model
pipeline = joblib.load("models/RF_v1.0.pkl")

# Loading the data
features = pd.read_parquet("data/inference_feature_example.parquet")

# Setting up the preprocessor
preproc = inferencePreprocessor(features)

# Preprocessing X
X = preproc.clean(return_df=True)

# Making Predictions
predictions = pipeline.predict(X)
