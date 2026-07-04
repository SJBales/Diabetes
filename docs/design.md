# Directory Structure

data --> holds script for downloading target and feature datasets from openml and parquet files
experiments --> holds the cross-validatio result tables from a couple of hyperparamter tuning experiments
models --> holds blob data of final trained model
notebooks --> contains notebooks for EDA and experimentation
src --> direcotry for production code
tests --> unit tests of production scripts

## Production Design (src)

### Helper Modules
exceptions.py --> project specific exceptions for meaningful error messages
logger_config.py --> script for a function to configure a project-specific logger

### Workhorse Modules

Each module is intended to contain key capabilities of the model to support production:
- "preprocessor.py" defines the preprocessors for preparing the data that is fed to pipelines
- "pipeline.py" creates a class for setting up the sklearn pipeline. These includ a column transformer and defining data preprocessing steps of encoding and scaling. Optimal hyperparamters are also defined in this module.
- "train.py" is a standalone module that (1) trains the model, (2) saves it in binary format to the models directory, and (3) examine test performance
- "predict.py" is a standalone module to load the model, data, and make predictions for new records. Ideal implementation is for overnight batch predictions of new diabetes patients for serving in a frontend UI for hospital staff to evaluate patients.

#### preprocessor.py

Purpose: 
Contains two classes to support preprocessing of data before it is fed to the pipeline: (1) trainingPreprocessor and (2) inferencePreprocessor. trainingPreprocessor has additional methods to give future flexibility in the training process, mostly for creating train and test splits which are not needed in production for inference.

Future Refinements:
Methods .create_train_test() and .clean() can be revamped. Controlling when a dataframe is returned from these modules is clunky and not intuitive. The suggested redesign is to have .clean() return the attributes of X_train, X_test, etc. instead of returning the results of train_test_split directly.

#### pipeline.py

Purpose:
Defines the class getPipeline with two methods: .get_column_transformer() which creates a column transformer that contains all preprocessing steps and .make_training_pipeline() which constructs an end to end proprocessing + training model pipeline.

Future Refinements:
Currently the module is only set up to train a RF model and will return an eror if any other model is selected. This can be expanded to accept other model arguments if model evaluation suggests they will be useful.

#### train.py

This module implements classes from the preprocessor.py and pipeline.py modules. It wraps the trainingPreproessor and getPipeline classes in one function so the module can be called standalone to preprocess data, train the model and evaluate its performance.

Future Refinements:
The module can be extended to handle retraining of the model. It's currently configured for training the model for initial deployment.

#### predict.py

Leverages the inferenceProcessor from the preprocessor.py. This is a lighter-weight, less feature rich preprocessor specifically set up for inference preprocessing as mentioned above. The module returns for predictions for the data loaded as part of the make_predictions function.

Future Refinements:
This module is very bare bones and only implemented for demonstration purposes. There is no data pipeline for this to be implemented into. The module needs to be expanded to read new patient data and write predictions back to the database with proper error-handling and type checking to prevent errors.
