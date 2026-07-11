import pytest
import pandas as pd


# Creating a raw data fixture
@pytest.fixture
def raw_input_features():
    return pd.DataFrame({
        'patient_nbr': [12341234, 23452345, 34563456, 45674567, 12341234],
        'age': ['[20-30)', '[30-40)', '[40-50)', '[30-40)', '[20-30)'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', '?', 'Caucasian'],
        'admission_source_id': [1, 3, 7, 12, 1],
        'discharge_disposition_id': [1, 2, 5, 6, 1],
        'A1Cresult': ['>8', '>7', 'Norm', 'None', '>8'],
        'max_glu_serum': ['None', '>300', 'Norm', '>200', 'None'],
        'insulin': ['No', 'Up', 'Down', 'Steady', 'Up'],
        'metformin': ['Up', 'No', 'Steady', 'Down', 'Up'],
        'weight': ['?', '[50-75)', '[0-25)', '[100-125)', '?'],
        'payer_code': ['?', 'MC', 'MD', 'HM', '?'],
        'medical_specialty': ['Pediatrics-Endocrinology',
                              '?',
                              'InternalMedicine',
                              'Family/GeneralPractice',
                              'Pediatrics-Endocrinology']
    })


@pytest.fixture
def raw_input_target():
    return pd.DataFrame({'readmitted': ['NO', '<30', '>30', 'NO', '<30']})


@pytest.fixture
def raw_input_features_train():
    return pd.DataFrame({
        'patient_nbr': [
            12341234, 23452345, 34563456, 45674567, 12341234,
            56785678, 67896789, 78907890, 89018901, 90129012,
            10111011, 11121112, 12131213, 13141314, 14151415,
            15161516, 16171617, 17181718, 18191819, 19201920,
            20212021, 21222122, 22232223, 23242324, 24252425,
            25262526, 26272627, 27282728, 28292829, 29302930,
        ],
        'age': [
            '[20-30)', '[30-40)', '[40-50)', '[30-40)', '[20-30)',
            '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[10-20)',
            '[40-50)', '[60-70)', '[70-80)', '[50-60)', '[30-40)',
            '[80-90)', '[60-70)', '[40-50)', '[70-80)', '[50-60)',
            '[90-100)', '[20-30)', '[60-70)', '[70-80)', '[50-60)',
            '[40-50)', '[80-90)', '[30-40)', '[60-70)', '[70-80)',
        ],
        'race': [
            'Caucasian', 'AfricanAmerican', 'Hispanic', '?', 'Caucasian',
            'Caucasian', 'Asian', 'Other', 'AfricanAmerican', 'Caucasian',
            '?', 'Caucasian', 'AfricanAmerican', 'Hispanic', 'Caucasian',
            'Caucasian', 'Other', 'AfricanAmerican', 'Caucasian', 'Asian',
            'Caucasian', 'Hispanic', 'Caucasian', 'AfricanAmerican', '?',
            'Caucasian', 'Caucasian', 'Other', 'AfricanAmerican', 'Caucasian',
        ],
        'admission_source_id': [
            1, 3, 7, 12, 1,
            7, 1, 4, 7, 2,
            1, 7, 6, 1, 7,
            4, 1, 7, 2, 1,
            7, 1, 5, 7, 1,
            3, 7, 1, 7, 4,
        ],
        'discharge_disposition_id': [
            1, 2, 5, 6, 1,
            1, 3, 1, 2, 1,
            6, 1, 2, 1, 5,
            1, 1, 3, 1, 6,
            1, 5, 1, 5, 1,
            1, 2, 1, 6, 1,
        ],
        'A1Cresult': [
            '>8', '>7', 'Norm', 'None', '>8',
            'None', 'None', '>8', 'Norm', 'None',
            '>7', 'None', 'None', '>8', 'None',
            'Norm', 'None', 'None', '>8', 'None',
            'None', '>7', 'None', 'Norm', 'None',
            '>8', 'None', 'None', '>7', 'None',
        ],
        'max_glu_serum': [
            'None', '>300', 'Norm', '>200', 'None',
            'None', 'None', 'Norm', 'None', '>200',
            'None', 'None', '>300', 'None', 'Norm',
            'None', 'None', '>200', 'None', 'None',
            'Norm', 'None', 'None', 'None', '>300',
            'None', 'Norm', 'None', 'None', 'None',
        ],
        'insulin': [
            'No', 'Up', 'Down', 'Steady', 'Up',
            'No', 'Steady', 'Up', 'No', 'Down',
            'Steady', 'No', 'Up', 'Steady', 'No',
            'Down', 'Steady', 'No', 'Up', 'Steady',
            'No', 'Up', 'Steady', 'Down', 'No',
            'Steady', 'Up', 'No', 'Steady', 'No',
        ],
        'metformin': [
            'Up', 'No', 'Steady', 'Down', 'Up',
            'No', 'No', 'Steady', 'No', 'No',
            'Steady', 'No', 'Down', 'No', 'Steady',
            'No', 'No', 'Up', 'No', 'Steady',
            'No', 'No', 'Steady', 'No', 'Down',
            'No', 'Steady', 'No', 'No', 'Up',
        ],
        'weight': [
            '?', '[50-75)', '[0-25)', '[100-125)', '?',
            '?', '?', '[75-100)', '?', '[50-75)',
            '?', '?', '[75-100)', '?', '?',
            '[100-125)', '?', '?', '[50-75)', '?',
            '?', '[25-50)', '?', '?', '[75-100)',
            '?', '?', '[50-75)', '?', '?',
        ],
        'payer_code': [
            '?', 'MC', 'MD', 'HM', '?',
            'MC', 'BC', '?', 'MC', 'SP',
            'HM', 'MC', '?', 'MD', 'MC',
            'BC', '?', 'MC', 'SP', 'HM',
            'MC', '?', 'MC', 'BC', 'MD',
            '?', 'MC', 'HM', 'MC', 'SP',
        ],
        'medical_specialty': [
            'Pediatrics-Endocrinology', '?', 'InternalMedicine',
            'Family/GeneralPractice', 'Pediatrics-Endocrinology',
            'Cardiology', '?', 'InternalMedicine', 'Surgery-General', '?',
            'Emergency/Trauma', 'InternalMedicine', '?', 'Cardiology',
            'Family/GeneralPractice',
            '?', 'InternalMedicine', 'Nephrology', '?', 'Cardiology',
            'InternalMedicine', '?', 'Emergency/Trauma', 'Surgery-General',
            '?',
            'InternalMedicine', 'Cardiology', '?', 'InternalMedicine',
            'Family/GeneralPractice',
        ],
    })


@pytest.fixture
def raw_input_target_train():
    return pd.DataFrame({'readmitted': [
        'NO', '<30', '>30', 'NO', '<30',
        'NO', '>30', '<30', 'NO', 'NO',
        '>30', 'NO', '<30', '>30', 'NO',
        '<30', 'NO', 'NO', '<30', '>30',
        'NO', '<30', 'NO', '>30', 'NO',
        '<30', 'NO', '>30', '<30', 'NO',
    ]})
