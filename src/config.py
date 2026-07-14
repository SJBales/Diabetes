NUMERIC_COLS = ['time_in_hospital',
                'num_lab_procedures',
                'num_procedures',
                'num_medications',
                'number_outpatient',
                'number_inpatient',
                'number_diagnoses']

CATEGORICAL_COLS = ['race',
                    'gender',
                    'admission_type_id',
                    'admission_source_id',
                    'discharge_disposition_id',
                    'A1Cresult',
                    'max_glu_serum',
                    'insulin',
                    'metformin',
                    'weight',
                    'payer_code',
                    'medical_specialty']

EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS
