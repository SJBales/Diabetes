from pipeline import getPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pytest
from exceptions import ModelError


def test_column_transformer():
    '''Validates that the ColumnTransformer was created'''

    # Initializing a pipeline class
    gp = getPipeline()

    # Testing the make column transformer method
    gp.make_column_transformer()

    assert gp.ct is not None
    assert isinstance(gp.ct, ColumnTransformer)


def test_make_pipeline():
    '''Validates that the pipeline was created'''

    # Initializing another pipeline class
    gp_pipe = getPipeline()

    gp_pipe.make_training_pipeline()

    # Testing the make_column_transformer method was run
    assert isinstance(gp_pipe.ct, ColumnTransformer)

    # Testing that the pipeline was created
    assert isinstance(gp_pipe.pipeline, Pipeline)


def test_invalid_input_raise():
    '''Validates the Model Error Exception is Raised'''

    input_gp = getPipeline()

    # Testing that the method should raise a Model Error Exception
    with pytest.raises(ModelError, match="Model requested is not supported"):
        input_gp.make_training_pipeline(model="Logistic Regression")
