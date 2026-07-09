import pandas as pd
from preprocessor import trainingPreprocessor

expected_fe_df = pd.DataFrame({
        'patient_nbr': [12341234, 23452345, 34563456, 45674567],
        'age': ['[20-30)', '[30-40)', '[40-50)]', '[30-40)]'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', '?'],
        'admission_source_id': [1, 3, 7, 12],
        'A1Cresult': ['>8', '>7', 'Norm', 'None'],
        'max_glu_serum': ['None', '>300', 'Norm', '>200'],
        'insulin': ['No', 'Up', 'Down', 'Steady'],
        'metformin': ['Up', 'No', 'Steady', 'Down'],
        'weight': ['?', '[50-75)', '[0-25)', '[100-125)'],
        'payer_code': ['?', 'MC', 'MD', 'HM'],
        'medical_specialty': [
            'Pediatrics-Endocrinology',
            '?',
            'InternalMedicine',
            'Family/GeneralPractice'],
        'target': [0, 1, 0, 0]
    })

expected_missing_df = pd.DataFrame({
        'patient_nbr': [12341234, 23452345, 34563456, 45674567],
        'age': ['[20-30)', '[30-40)', '[40-50)]', '[30-40)]'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', '?'],
        'admission_source_id': [1, 3, 7, 12],
        'A1Cresult': ['>8', '>7', 'Norm', 'None'],
        'max_glu_serum': ['None', '>300', 'Norm', '>200'],
        'insulin': ['No', 'Up', 'Down', 'Steady'],
        'metformin': ['Up', 'No', 'Steady', 'Down'],
        'weight': ['?', '[50-75)', '[0-25)', '[100-125)'],
        'payer_code': ['?', 'MC', 'MD', 'HM'],
        'medical_specialty': [
            'Pediatrics-Endocrinology',
            '?',
            'InternalMedicine',
            'Family/GeneralPractice'],
        'target': [0, 1, 0, 0],
        'missing_insulin': [1, 0, 0, 0],
        'missing_metformin': [0, 1, 0, 0],
        'metformin_change': [1, 0, 0, -1],
        'insulin_change': [0, 1, -1, 0],
        'missing_weight': [1, 0, 0, 0],
        'payer_code_cleaned': ['missing_payer', 'MC', 'MD', 'HM'],
        'medical_specialty_cleaned': [
            'Pediatrics-Endocrinology',
            'missing_medical_specialty',
            'InternalMedicine',
            'Family/GeneralPractice'],
        'missing_a1c': [0, 0, 0, 1],
        'missing_max_glu_serum': [1, 0, 0, 0]
})


# Writing a test for the trainingProcessor
def test_training_processor(raw_input_features, raw_input_target):

    # Testing the first_encounter method of the trainingPreprocessor class
    processor = trainingPreprocessor(raw_input_features, raw_input_target)
    processor.first_encounter()

    pd.testing.assert_frame_equal(
        processor.split_df.reset_index(drop=True),
        expected_fe_df)

    # Testing the method to handle missing values
    processor.handle_missings()

    pd.testing.assert_frame_equal(
        processor.split_df.reset_index(drop=True),
        expected_missing_df
    )

    # Testing the method for creating X and y dataframes
    processor.create_X_y()

    pd.testing.assert_series_equal(
        processor.y.reset_index(drop=True),
        expected_missing_df['target'],
        check_names=False
    )

    assert 'patient_nbr' not in processor.X.columns.values
    assert 'target' not in processor.X.columns.values
    assert 'max_glu_serum' not in processor.X.columns.values
    assert len(processor.X) == len(processor.y)

    # Testing the method for creating train and test splits
    processor.create_train_test()

    assert processor.X_train is not None
    assert processor.X_test is not None
    assert processor.y_train is not None
    assert processor.y_test is not None
    assert len(processor.X_train) == len(processor.y_train)
    assert len(processor.X_test) == len(processor.y_test)
    assert len(processor.X_train) != len(processor.X_test)

    # Testing the master .clean() method
    master_processor = trainingPreprocessor(raw_input_features,
                                            raw_input_target)

    master_processor.clean()

    print(master_processor.X_train, processor.X_train)
    pd.testing.assert_frame_equal(master_processor.X_train,
                                  processor.X_train)
    pd.testing.assert_frame_equal(master_processor.X_test,
                                  processor.X_test)
    pd.testing.assert_series_equal(master_processor.y_train,
                                   processor.y_train)
    pd.testing.assert_series_equal(master_processor.y_test,
                                   processor.y_test)
