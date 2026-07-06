import pytest
import pandas as pd
from preprocessor import trainingPreprocessor

expected_df = pd.DataFrame({
        'patient_nbr': [12341234, 23452345, 34563456, 45674567],
        'age': ['[20-30)', '[30-40)', '[40-50)]', '[30-40)]'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', '?'],
        'admission_source_id': [1, 3, 7, 12],
        'A1Cresult': ['>8', '>7', 'Norm', 'None'],
        'insulin': ['No', 'Up', 'Down', 'Steady'],
        'target': [0, 1, 0, 0]
    })


# Writing a test for the trainingProcessor
def test_training_processor(raw_input_features, raw_input_target):

    processor = trainingPreprocessor(raw_input_features, raw_input_target)
    processor.first_encounter()

    pd.testing.assert_frame_equal(
        processor.split_df.reset_index(drop=True),
        expected_df)
