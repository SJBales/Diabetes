# 30-Day Readmission For Diabetes Patients

# 1. Problem Statement & Goals

## Business Impact

Hospital readmission within 30 days is a highly negative outcome for patients. A high 30-day readmission rate also affects Medicare reimbursement rates as it is viewed as a marker of lower quality care. Understanding which patients are higher risk at discharge can help guide risk-management practices before discharge and identify patients for proactive outreach once released.

## Goals

This project has two goals:
- Identify which features indicate patients are at higher risk of readmission for risk management ahead of discharge
- Predict which patients are highest risk of readmission for mitigation measures post discharge, such as frequent follow-up calls and check-in visits

# 2. Dataset

This project uses the DiabetesUS130 dataset from openML that contains the readmission status (target) and readmission risk factors (features) of diabetes patients at 130 US hospitals. 

# 3. Methodology

## Evaluation Criteria

Flagging patients as likely to readmit that don't (false positives) has a lower overall impact than missing patients that are likely to readmit (false negatives). False negatives cannot be fully optimized without considering false positives as it will add notification fatigue to the staff and impact their resourcing due to increased patient outreach. Recall was used to account for the relative greater impact of false negatives vs. false positives, but precision and average precision were included as evaluation metrics to balance the impact of false positives.

# 4. Results

Final performance of the random forest classifier on the heldout test set (~15K observations) is:

- Recall: 60.4%
- Precision: 11.49%

# 5. Running the Code

## Setup

All dependency module requirements are captured in requirements.txt. To run the code contained in the project, first set up a python virtual environment and run the command pip install -r requirements.txt to install all package dependencies.

## Inference Predictions

Making inference predictions is self-contained in src/train.py. Run the script using the command python3 src/predict.py.

## Training

Model training steps are implemented in the src/train.py script. Retrain the model using the command python3 src/train.py.

Note: optimal hyperparameters are hard-coded for consistency over time. The wrapper function can be modified to accept an an argument or hyperparameters can be directly modified if needed. However, this is only recommended after drifts in production have been observed and additional experimentation has been conducted to reassess optimal hyperparameters. 

# 6. Limitations and Next Steps

## Limitations

The dataset has a few limitations: 

1. Some patients have several encounters but there is no timestamp associated with the encounter to yield a definitive order. Treatment history and patterns of the patient are likely useful for predicting future readmissin, but with no definitive indicator of visit order the observations were limited to the first encounter. This is the most conservative and robust way of handling multiple encounters but information is certainly being lost (~25K rows)
2. Manual sparse features were included in the original dataset that were not included as features in the final model due to low feature importance
3. Key data that is easy to collect (e.g. weight) and likely in other EHR database is missing

## Next Steps

The dataset used in this project is static, with no direct interface to the business users to act on the predictions made from the model. Displaying the predictions in any existing technology nurses or other hopsital admin are using is an important next step to serve the predictions to the business users. Before that, the data pipeline and predictions will need to be refactored to handle live, incremental production data as adittional patients and admitted and discharged from hospitals. An important consideration is how often predictions should be made. Business users do not need real time predictions, and the likelihood of readmission will not change much once the features have been observed. Therefore, I recommend nightly batch processing of predictions of only newly discharged patients instead of configuring streaming predictions.