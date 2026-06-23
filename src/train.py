from preprocessor import load_data, trainingPreprocessor
from pipeline import getPipeline
import logging

# Setting up the logger
from logger_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

# ------------------ Loading Data ------------------

# Loading the data
features, target = load_data()

# ------------------ Running the Preprocessor ------------------

# Setting up the training preprocessor
preprocessor = trainingPreprocessor(features, target)

# Running the cleaning steps
X_train, X_test, y_train, y_test = preprocessor.clean(return_df=True)

# ------------------ Training the Model ------------------

# Defining optimal parameters
optimal_hypers = {
    'max_depth': 5,
    'min_samples_split': 10,
    'n_estimators': 200,
    'max_features': 0.6,
    'class_weight': 'balanced'
}

# Setting up the pipeline
training_pipeline = getPipeline().make_training_pipeline(model="RF",
                                                         return_pipeline=True,
                                                         **optimal_hypers)

logger.info("Successfully created training pipeline")

# Fitting the pipeline
training_pipeline.fit(X_train, y_train)

logger.info("Successfully fit model on training data")
