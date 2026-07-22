import pandas as pd
import logging
import warnings

# Sklearn Imports
from sklearn.model_selection import train_test_split

# Local module imports
from exceptions import NotProcessedError, DataModelError
from logger_config import setup_logger
from config import DROP_COLS, NUMERIC_COLS, ONEHOT_COLS, ORDINAL_COLS, TARGET_COL

# Setting up the logger
setup_logger()
logging.captureWarnings(True)

logger = logging.getLogger(__name__)


# Function for loading data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Function to load both target and features from the 
    local data'''

    features = pd.read_parquet("data/diabetes_features.parquet")
    target = pd.read_parquet("data/diabetes_target.parquet")

    logger.info("Successfully Loaded Data")

    return features, target


# Defining a class for handling all preprocessing steps
class trainingPreprocessor:
    '''Class to Create a Training Preprocessor'''

    def __init__(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self.raw_features = features
        self.raw_target = target
        self.split_df = None
        self.X = None
        self.y = None
        self._missing_processed_ = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def first_encounter(self) -> None:
        '''Limits the input dataframe to the first enounter of the patient'''

        # Creating a dataframe for selecting first encounters
        splitting_df = self.raw_features.copy()
        splitting_df['target'] = (self.raw_target['readmitted'] == "<30").astype(int)

        # Keeping only the first encounter per patient_nbr
        self.split_df = splitting_df.drop_duplicates('patient_nbr',
                                                     keep='first')

        logger.info("Successfully subset to the first encounter")

    def handle_missings(self) -> None:
        '''Handles preprocessing of missing data'''

        # Checking that the split DF has been created
        if self.split_df is None:
            logger.info("Data has not been limited to first encounter, running method")

            self.first_encounter()

        # Checking that split DF has the correct columns
        df_cols = set(self.split_df.columns)
        expected_cols = set(NUMERIC_COLS + ONEHOT_COLS + ORDINAL_COLS + DROP_COLS + TARGET_COL)

        missing_cols = expected_cols - df_cols
        extra_cols = df_cols - expected_cols

        if missing_cols or extra_cols:
            raise DataModelError(f"missing={sorted(missing_cols)} unexpected={sorted(extra_cols)}")

        # Creating a feature to indicate if a patient is not on insulin or metformin
        self.split_df['missing_insulin'] = (self.split_df.insulin == "No").astype(int)
        self.split_df['missing_metformin'] = (self.split_df.metformin == "No").astype(int)

        # Creating a dictionary to encode changes in insulin and metformin
        encode_dict = {'No': 0,
                       "Steady": 0,
                       "Up": 1,
                       "Down": -1}

        # Applying the encoding to engineer new features
        self.split_df['metformin_change'] = self.split_df['metformin'].apply(lambda x: encode_dict[x])
        self.split_df['insulin_change'] = self.split_df['insulin'].apply(lambda x: encode_dict[x])

        # Encoding Missing Weight Values
        self.split_df['missing_weight'] = (self.split_df['weight'] == "?").astype("int")

        # Utility function for encoding missing values in a column for one-hot encoding
        def missing_cleaner(x, missing_code: str, encoding="") -> None:
            '''Helper function for cleaning columns with specific strings
            encoding missing values'''
            if x == missing_code:
                return encoding
            else:
                return x

        # Encoding Missing Payer Code and Medical Specialty
        self.split_df['payer_code_cleaned'] = self.split_df['payer_code']\
            .apply(missing_cleaner, args=("?", "missing_payer"))
        self.split_df['medical_specialty_cleaned'] = self.split_df['medical_specialty']\
            .apply(missing_cleaner, args=("?", "missing_medical_specialty"))

        # Encoding Missing A1C and Max Glucose Serum
        self.split_df['missing_a1c'] = (self.split_df['A1Cresult'] == "None").astype("int")
        self.split_df['missing_max_glu_serum'] = (self.split_df['max_glu_serum'] == "None").astype("int")

        # Updating attribute to flag processing has been completed
        self._missing_processed_ = True

        # Logging completed work
        logger.info("Successfully processed missing data")

    def create_X_y(self) -> None:
        '''Creates X and y dataframes for modeling'''

        # Checking that missing data has been processed
        if not self._missing_processed_:
            warnings.warn("Missing data has not been processed, running method")
            self.handle_missings()

        # List of features to drop
        drop_features = DROP_COLS + TARGET_COL

        # Building X & y dataframes
        self.y = self.split_df['target']
        self.X = self.split_df.drop(columns=drop_features, errors="ignore")

        # Building X & y dataframes
        self.y = self.split_df['target']
        self.X = self.split_df.drop(columns=drop_features, errors="ignore")

        # Logging completed work
        logger.info("Successfully created X & y dataframes")

    # Method for creating train and test splits
    def create_train_test(self, return_df=False, test_size_=0.2) -> pd.DataFrame | None:
        '''Creates train and test splits for modeling'''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                random_state=42,
                                                                                test_size=test_size_)

        if return_df:
            logger.info("Returning train and test splits of X & y")
            return self.X_train, self.X_test, self.y_train, self.y_test

    def clean(self, return_df=False, test_size_=0.2) -> pd.DataFrame | None:
        '''Master method for running all cleaning methods'''

        self.first_encounter()
        self.handle_missings()
        self.create_X_y()
        self.create_train_test(test_size_=test_size_)

        if return_df:
            logger.info("Returning train and test splits of X & y")
            return self.X_train, self.X_test, self.y_train, self.y_test

        # Logging status
        logger.info("Successfully completed all cleaning steps")


class inferencePreprocessor:
    '''Creating the class to handle preprocessing for inference'''

    def __init__(self, features) -> None:
        self.raw_features = features
        self.missing_processed_ = False
        self.split_df = None
        self.X = None

    def first_encounter(self) -> None:
        '''Limits the input dataframe to the first enounter of the patient'''

        # Keeping only the first encounter per patient_nbr
        self.split_df = self.raw_features.drop_duplicates('patient_nbr',
                                                          keep='first')

    def handle_missings(self) -> None:
        '''Handles preprocessing of missing data'''

        # Checking that the split DF has been created
        if self.split_df is None:
            logger.info("Data has not been limited to first encounter, running method")

            self.first_encounter()

        # Checking that split DF has the correct columns
        df_cols = set(self.split_df.columns)
        expected_cols = set(NUMERIC_COLS + ONEHOT_COLS + ORDINAL_COLS + DROP_COLS)

        missing_cols = expected_cols - df_cols
        extra_cols = df_cols - expected_cols

        if missing_cols or extra_cols:
            raise DataModelError(f"missing={sorted(missing_cols)} unexpected={sorted(extra_cols)}")

        # Creating a feature to indicate if a patient is not on insulin or metformin
        self.split_df['missing_insulin'] = (self.split_df.insulin == "No").astype(int)
        self.split_df['missing_metformin'] = (self.split_df.metformin == "No").astype(int)

        # Creating a dictionary to encode changes in insulin and metformin
        encode_dict = {'No': 0,
                       "Steady": 0,
                       "Up": 1,
                       "Down": -1}

        # Applying the encoding to engineer new features
        self.split_df['metformin_change'] = self.split_df['metformin'].apply(lambda x: encode_dict[x])
        self.split_df['insulin_change'] = self.split_df['insulin'].apply(lambda x: encode_dict[x])

        # Encoding Missing Weight Values
        self.split_df['missing_weight'] = (self.split_df['weight'] == "?").astype("int")

        # Utility function for encoding missing values in a column for one-hot encoding
        def missing_cleaner(x, missing_code, encoding=""):
            if x == missing_code:
                return encoding
            else:
                return x

        # Encoding Missing Payer Code and Medical Specialty
        self.split_df['payer_code_cleaned'] = self.split_df['payer_code']
        self.split_df['medical_specialty_cleaned'] = self.split_df['medical_specialty']\
            .apply(missing_cleaner, args=("?", "missing_medical_specialty"))

        # Encoding Missing A1C and Max Glucose Serum
        self.split_df['missing_a1c'] = (self.split_df['A1Cresult'] == "None").astype("int")
        self.split_df['missing_max_glu_serum'] = (self.split_df['max_glu_serum'] == "None").astype("int")

        self.missing_processed_ = True

    def create_X(self, return_df=False) -> pd.DataFrame | None:
        '''Creates X dataframe for modeling'''

        # Checking that missing data has been processed for production, raises exception if not
        if not self.missing_processed_:
            raise NotProcessedError("Missing data has not been processed")

        # List of features to drop
        drop_features = DROP_COLS

        # Building X & y dataframes
        self.X = self.split_df.drop(columns=drop_features, errors="ignore")

        self.X = self.split_df.drop(columns=drop_features)

        # Returning the dataframe
        if return_df:
            return self.X

    def clean(self, return_df: bool) -> pd.DataFrame | None:
        '''Master method for running all cleaning methods'''

        try:
            self.first_encounter()
            self.handle_missings()
            self.create_X()

            if return_df:
                return self.X

        except NotProcessedError:
            logger.exception("Pipeline step called out of order")
            raise
