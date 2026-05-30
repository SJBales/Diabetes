# Analysis Objectives 

## Business Impact

Hospital readmission within 30 days are a negative outcome for patients. A high 30-day readmission rate also affects Medicare reimbursement rates. Understading which patients are higher risk at discharge can help guide risk-management practices before discharge and identify patients for proactive outreach once released.

## Goals

This project has two goals:
- Identify which factors indicate patients are at higher risk of readmission for risk management ahead of discharge
- Predict which patients are highest risk of readmission for mitigation measures post discharge, such as frequent follow-up calls and check-in visits.

# Analysis Plan

## Steps

Predictive Model:
- Exploratory analysis to understand outliers, distributions and outliers
- Divide into train, validation and test splits
- Create pipeline for cleaning and preprocessing with the following steps:
  - Handle missing data
  - Scale numeric features and encode categorical features
- Engineer feature
- Fit, evaluate and compare models; select features
- Assess feature importance and impact to socialize with stakeholders for proactive risk management
- Package the post release model for deployment
