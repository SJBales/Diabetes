from preprocessor import load_data, trainingPreprocessor
from pipeline import getPipeline

# Loading the data
features, target = load_data()

# Setting up the training preprocessor
preprocessor = trainingPreprocessor(features, target)

# Running the cleaning steps
X_train, X_test, y_train, y_test = preprocessor.clean()

# Defining optimal parameters
optimal_hypers = {
    'max_depth': 5,
    'min_samples_split': 10,
    'n_estimators': 200,
    'max_features': 0.6,
    'class_weight': 'balanced'
}

# Setting up the pipeline
training_pipeline = getPipeline(optimal_hypers,
                                model="RF",
                                return_pipeline=True)
