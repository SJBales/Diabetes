import pandas as pd

def load_data():

    features = pd.read_parquet("data/diabetes_features.parquet")
    target = pd.read_parquet("data/diabetes_target.parquet")
    
    return features, target


class dataPreprocessor:

    def __init__(self, features, target) -> None:
        self.raw_fetaures = features
        self.raw_target = target
        self.split_df = None
        self.X = None
        self.y = None

    def first_encounter(self) -> None:
        '''Limits the input dataframe to the first enounter of the patient'''

        # Creating a dataframe for selecting first encounters
        splitting_df = features.copy()
        splitting_df['target'] = (target['readmitted'] == "<30").astype(int)

        # Keeping only the first encounter per patient_nbr
        self.split_df = splitting_df.drop_duplicates('patient_nbr', keep='first')

    def handle_missings(self) -> None:
        '''Handles preprocessing of missing data'''

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
        self.split_df['payer_code_cleaned'] = self.split_df['payer_code']\
            .apply(missing_cleaner, args=("?", "missing_payer"))
        self.split_df['medical_specialty_cleaned'] = self.split_df['medical_specialty']\
            .apply(missing_cleaner, args=("?", "missing_medical_specialty"))

        # Encoding Missing A1C and Max Glucose Serum
        self.split_df['missing_a1c'] = (self.split_df['A1Cresult'] == "None").astype("int")
        self.split_df['missing_max_glu_serum'] = (self.split_df['max_glu_serum'] == "None").astype("int")

    def create_train_test(self):
        # List of features to drop
        drop_features = ['patient_nbr', 'encounter_id', 'target',
                        'repaglinide', 'nateglinide', 'chlorpropamide',
                        'glimepiride', 'acetohexamide', 'glipizide',
                        'glyburide', 'tolbutamide',
                        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                        'tolazamide', 'examide', 'citoglipton', 'insulin',
                        'glyburide.metformin', 'glipizide.metformin',
                        'glimepiride.pioglitazone', 'metformin.rosiglitazone',
                        'metformin.pioglitazone', 'metformin', 'insulin',
                        'A1Cresult', 'weight', 'diag_1', 'diag_2', 'diag_3',
                        'max_glu_serum', 'change']

        # Building X & y dataframes
        self.y = self.split_df['target']
        self.X = self.split_df.drop(columns=drop_features)

        # Creating train and test splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                random_state=42,
                                                                                test_size=0.2)
    def clean(self):
        '''Master method for running all cleaning methods'''
        self.first_encounter()
        self.handle_missings()
        self.create_train_test()

if __name__ == "__main__":
    features, target = load_data()

    preprocessor_engine = dataPreprocessor()

    dataPreprocessor.clean()