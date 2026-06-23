# Directory Structure

data --> holds script for downloading data from openml and parquet files for target and features
experiments --> holds the CV results from various HP tuning experiments
models --> holds blob data of trained model
notebooks --> contains notebooks for EDA and experimentation
src --> production code
tests --> unit tests of production scripts

## Production Design (src)

### Helper Modules
exceptions.py --> project specific exceptions for meaningful error messages
logger_config.py --> script for a function to configure a project-specific logger

### Workhorse Modules

Each module is intended to contain key capabilities of the model to support production:
- "preprocessor.py" defines the preprocessors for preparing the data that is fed to pipelines
- "pipeline.py" creates a class for setting up the sklearn pipeline including a column transformer and defining data preprocessing steps of encoding and scaling. This script is also where the optimal hyperparameters are defined. 
- "train.py" is a standalone scrip to train the model, save it in binary format to the models directory and examine test performance

#### preprocessor.py

Purpose: 
Contains two classes to support preprocessing of data before it is fed to the pipeline: trainingPreprocessor and inferencePreprocessor. trainingPreprocessor has more methods, mostly for creating train and test splits which are not needed in production for inference.

Refinements:
Methods .create_train_test() and .clean() need revamped. Controlling when a dataframe is returned from these modules is clunky and not intuitive. The suggested redesign is to have .clean() return the attributes of X_train, X_test, etc. instead of returning the results of train_test_split directly.

