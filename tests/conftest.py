import pytest
import pandas as pd


# Creating a raw data fixture
@pytest.fixture
def raw_input_features():
    return pd.DataFrame({
        'patient_nbr': [12341234, 23452345, 34563456, 45674567, 12341234],
        'age': ['[20-30)', '[30-40)', '[40-50)]', '[30-40)]', '[20-30)'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', '?', 'Caucasian'],
        'admission_source_id': [1, 3, 7, 12, 1],
        'A1Cresult': ['>8', '>7', 'Norm', 'None', '>8'],
        'insulin': ['No', 'Up', 'Down', 'Steady', 'Up']
    })

@pytest.fixture
def raw_input_target():
    return pd.DataFrame({'readmitted': ['NO', '<30', '>30', 'NO', '<30']})
