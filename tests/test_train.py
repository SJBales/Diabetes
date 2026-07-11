# Plan
'''
Utilize existing data fixture to test preprocessor and transformer
Create new validation dataset for post pipeline fitting
'''
import joblib
from preprocessor import trainingPreprocessor
from train import train_model
from sklearn.pipeline import Pipeline


def test_train_model_produces_usable_artifact(
    tmp_path, monkeypatch, raw_input_features_train, raw_input_target_train
):
    monkeypatch.setattr(
        "train.load_data",
        lambda: (raw_input_features_train, raw_input_target_train),
    )

    out = tmp_path / "RF_test.pkl"
    pipeline = train_model(
        model_path=out,
        hypers={"n_estimators": 5, "max_depth": 2},
    )

    assert isinstance(pipeline, Pipeline)
    assert out.exists()

    # Reproduce the same cleaned split the model was trained on
    prep = trainingPreprocessor(raw_input_features_train,
                                raw_input_target_train)
    _, X_test, _, y_test = prep.clean(return_df=True)

    loaded = joblib.load(out)
    preds = loaded.predict(X_test)

    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
