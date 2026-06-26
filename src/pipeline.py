from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
import logging

# Local module imports
from exceptions import ModelError
from logger_config import setup_logger

# Setting up the logger
setup_logger()
logging.captureWarnings(True)

logger = logging.getLogger(__name__)


class getPipeline():

    def __init__(self):
        self.OHEFEATURES = ['race', 'gender', 'admission_type_id',
                            'discharge_disposition_id', 'admission_source_id',
                            'payer_code', 'medical_specialty']
        self.SCALING_FEATURES = ['time_in_hospital', 'num_lab_procedures',
                                 'num_procedures', 'num_medications',
                                 'number_outpatient', 'number_inpatient',
                                 'number_diagnoses']
        self.ORDINAL_COLS = ['age']
        self.ct = None
        self.pipeline = None

    def make_column_transformer(self):
        '''Method for creating the column tranformer to use
           in a training and fitting pipeline'''
        # Pipeline for OHE
        ohe_pipeline = Pipeline([('handle_missing', SimpleImputer(strategy='most_frequent')),
                                ('encoding', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01, sparse_output=False))])

        # Pipeline for Scaling
        scaling_pipeline = Pipeline([('scaling', MinMaxScaler())])

        # Creating a list of categories for the age column
        age_categories = ['[0-10)',
                          '[10-20)',
                          '[20-30)',
                          '[30-40)',
                          '[40-50)',
                          '[50-60)',
                          '[60-70)',
                          '[70-80)',
                          '[80-90)',
                          '[90-100)']

        # Setting up a pipeline for ordinal encoding
        ordinal_pipeline = Pipeline([('ordinal_encoding',
                                    OrdinalEncoder(
                                        categories=[age_categories]
                                        ))])

        # Configuring the Column Transformer with the pipelines
        ct = ColumnTransformer([('OHE',
                                 ohe_pipeline,
                                 self.OHEFEATURES),
                                ('scaling',
                                 scaling_pipeline,
                                 self.SCALING_FEATURES),
                                ('ordinal_encoding',
                                 ordinal_pipeline,
                                 self.ORDINAL_COLS)])

        # Setting the output to return a pandas dataset
        ct.set_output(transform='pandas')

        # Assigning the column transformer as an attribute
        self.ct = ct

        # Logging
        logger.info("Successfully Initiatilzed Column Transformer")

    def make_training_pipeline(self,
                               model="RF",
                               return_pipeline=True,
                               **params):
        '''Method to create the end-to-end training pipeline for an RF mdoel'''

        # Setting up the column transformer
        if self.ct is None:
            logger.info("No column transformer, initializing")
            self.make_column_transformer()

        # Validating the use of RF
        if model != "RF":
            logger.exception(f"Model {model} requested is not supported")
            raise ModelError("Model requested is not supported")

        model_ = RandomForestClassifier(**params)

        self.pipeline = Pipeline([('preprocessing', self.ct),
                                  ('model', model_)])

        if return_pipeline:
            return self.pipeline


def log_performance(pipeline, X, y_actual, kind="TRAINING"):
    '''Method for determining performance on the test set'''
    preds = pipeline.predict(X)

    recall, average_precision = recall_score(y_actual, preds), precision_score(y_actual, preds, zero_division=0)
    logger.info(f"""------{kind}-------\n
                Recall is {round(recall, 4)}.\n
                Precision is {round(average_precision, 4)}""")
