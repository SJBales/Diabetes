from preprocessor import load_data, trainingPreprocessor
from pipeline import getPipeline, log_performance
import logging
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

# Setting up the logger
from logger_config import setup_logger
from exceptions import DataModelError
from config import NUMERIC_COLS, ONEHOT_COLS, ORDINAL_COLS, ENGINEERED_COLS, CLEANED_COLS

setup_logger()
logger = logging.getLogger(__name__)


def train_model(
    model_path: Path = Path("models/RF_v1.0.pkl"),
    hypers: dict | None = None,
) -> Pipeline:
    """Train the model and persist it."""
    if hypers is None:
        hypers = {
            'max_depth': 5,
            'min_samples_split': 10,
            'n_estimators': 200,
            'max_features': 0.6,
            'class_weight': 'balanced',
        }

    features, target = load_data()
    preprocessor = trainingPreprocessor(features, target)
    X_train, X_test, y_train, y_test = preprocessor.clean(return_df=True)

    # Checking that split DF has the correct columns
    df_cols = set(X_train.columns)
    expected_cols = set(NUMERIC_COLS + ONEHOT_COLS + ORDINAL_COLS + ENGINEERED_COLS + CLEANED_COLS)

    missing_cols = expected_cols - df_cols
    extra_cols = df_cols - expected_cols

    if missing_cols or extra_cols:
        raise DataModelError(f"missing={sorted(missing_cols)} unexpected={sorted(extra_cols)}")

    training_pipeline = getPipeline().make_training_pipeline(
        model="RF", return_pipeline=True, **hypers
    )
    training_pipeline.fit(X_train, y_train)

    log_performance(training_pipeline, X_train, y_train)
    log_performance(training_pipeline, X_test, y_test, kind="TEST")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_pipeline, model_path)

    return training_pipeline


if __name__ == "__main__":
    train_model()
