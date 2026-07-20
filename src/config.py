NUMERIC_COLS = ['time_in_hospital',
                'num_lab_procedures',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_inpatient',
                'number_diagnoses']

ONEHOT_COLS = ['race',
               'gender',
               'admission_type_id',
               'admission_source_id',
               'discharge_disposition_id']

CLEANED_COLS = ['payer_code_cleaned',
                'medical_specialty_cleaned']

ORDINAL_COLS = ['age']

ENGINEERED_COLS = ['missing_insulin',
                   'missing_metformin',
                   'metformin_change',
                   'insulin_change',
                   'missing_weight',
                   'missing_a1c',
                   'missing_max_glu_serum']

DROP_COLS = ['patient_nbr',
             'encounter_id',
             'diabetesMed',
             'number_emergency',
             'repaglinide',
             'nateglinide',
             'chlorpropamide',
             'glimepiride',
             'acetohexamide',
             'glipizide',
             'glyburide',
             'tolbutamide',
             'pioglitazone',
             'rosiglitazone',
             'acarbose',
             'miglitol',
             'troglitazone',
             'tolazamide',
             'examide',
             'citoglipton',
             'insulin',
             'glyburide.metformin',
             'glipizide.metformin',
             'glimepiride.pioglitazone',
             'metformin.rosiglitazone',
             'metformin.pioglitazone',
             'metformin',
             'insulin',
             'A1Cresult',
             'weight',
             'diag_1',
             'diag_2',
             'diag_3',
             'max_glu_serum',
             'change',
             'medical_specialty',
             'payer_code']

TARGET_COL = ['target']
