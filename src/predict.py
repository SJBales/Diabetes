import joblib
from preprocessor import inferencePreprocessor
import pandas as pd
import numpy as np


def make_predictions() -> np.array:
    '''Wrapper function to load model, data, set up preprocessor
    and make predictions'''
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

    return predictions


if __name__ == "__main__":
    preds = make_predictions()
    print(f"Predictions: {preds}")
