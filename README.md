# 30-Day Readmission For Diabetes Patients

Identifying key contributors to diabetes patient readmission within 30 days and training a ML model to be deployed to predict which patients are most at risk for mitigation.

# Problem Statement & Goals

## Business Impact

Hospital readmission within 30 days are a negative outcome for patients. A high 30-day readmission rate also affects Medicare reimbursement rates. Understading which patients are higher risk at discharge can help guide risk-management practices before discharge and identify patients for proactive outreach once released.

## Goals

This project has two goals:
- Identify which features indicate patients are at higher risk of readmission for risk management ahead of discharge
- Predict which patients are highest risk of readmission for mitigation measures post discharge, such as frequent follow-up calls and check-in visits

# Dataset

This project uses the DiabetesUS130 dataset from openML. 

# Methodology

## Evaluation Criteria

Flagging patients as likely to readmit that don't (false positives) has a lower overall impact than missing patients that are likely to readmit (false negatives). False negatives cannot be fully optimized without considering false positives as it will add notification fatigue to the staff and impact their resourcing due to increased patient outreach. I will use recall to account for the relative greater impact of false negatives vs. false positives, but also monitor precision and average precision to balance false positives.

## Analysis Plan -- Steps

Predictive Model:
- Exploratory analysis to understand outliers, distributions and outliers --> complete
- Divide into train, validation and test splits
- Create pipeline for cleaning and preprocessing with the following steps:
  - Handle missing data
  - Scale numeric features and encode categorical features
- Engineer features
- Fit, evaluate and compare models; select features
- Assess feature importance and impact to socialize with stakeholders for proactive risk management
- Package the post release model for deployment

# Results

# Limitations and Next Steps

## Limitations

The dataset has a few limitations: 

1. Some patients have several encounters, but there is no timestamp associated with the encounter to yield a definitive order. The first encounter based on encounter ID was retained for this project while the others were dropped. This is the most conservative and robust way of handling multiple encounters but information is certainly being lost (~25K rows)
2. Several irrelevant features were included
3. Key patient data that is easy to collect is missing like weight

## Next Steps