# customer_churn

Notebooks  contains betting_customer_churn.ipynb
main.py can also be used to run the full training pipeline.
We have used two methodology to solve the problem

Churning is defined as zero total turnover(sports and game) for 30 days.
 Mark a customer as churned if they have not placed a bet for a chosen period (30 days).

## Transactional context

We treat the features as flat transactions and have defined churning interval for 7 days.
This approach leverages daily or more granular data and flags inactivity directly in the raw transaction logs. Since the data is highly imbalanced, we apply SMOTE to undersample the non churned featuers and oversample the churned features.

New features: bet_frequency_ratio, std(total_turnover),  mean(total turnover)  and ratio of   std(total_turnover) with mean(total_turnover                              

Since the data is highly imbalanced, we apply SMOTE to undersample the non churned featuers and oversample the chruned features.


The features are trained with XGBoost classifier.


Results (Transactional)
       
|         | False | True  |
|---------|-------|-------|
| **False** | 52632 | 6326  |
| **True**  | 133   | 58806 |

F1  Score (Churn=False) = 0.94
F1  Scrore(Churn=True) = 0.95
              precision    recall  f1-score   support

       False       1.00      0.89      0.94     58958
        True       0.90      1.00      0.95     58939

    accuracy                           0.95    117897
   macro avg       0.95      0.95      0.95    117897
weighted avg       0.95      0.95      0.95    117897



## Resampled Time-Series context 

Aggregate player activity into weekly (or monthly) time windows.
Some of features are summed over week window but churning Aggregate is treated as max so 1 frame of churn in a week interval makes the 
window as churned.

We have added rolling_mean_turnover_4w(mean of rolling total turn over last 4 weeks), turnover_change(Fractional change between the total turnover)

We have  carried out time-based data Splitting to test this so first 70 %of temporal data is used for training and rest 30% for testing.

The data is highly imbalanced but we have avoided using Syntheitc data(SMOTE) but given the feautres are aggregate, we dont suffer from 
imbalanced classifier.


The features are trained with XGBoost classifier.


Results

AUC-ROC: 0.9956511782022617
|         | False | True  |
|---------|-------|-------|
| **False** | 183039 | 437  |
| **True**  | 481   | 2658 |

              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00    183476
         1.0       0.86      0.85      0.85      3139

    accuracy                           1.00    186615
   macro avg       0.93      0.92      0.93    186615
weighted avg       1.00      1.00      1.00    186615




## Ideal Implementation
1. Truth is that everyone churns at some time. So we need to reformulate this problem as to find when customer churns.
Can we define Churning as regression problem. Define churning in days when the user would leave the platform.
Solve it with survival analysis but there are two problems for labelling the data.
a. If we simply use the current tenure for customers who haven't churned, the model will be biased.
b. Completely excluding customers who haven't churned is also problematic

The standard approach is to solve it with Cox Proportional Hazards Model.


## Data gap analysis 
We lack gaming & customer behavior specific features as we come to conclusion that 
some players are churning away without seeing any major change in the feature patterns.
for e.g. Session time(time spent playing games/betting) is one of the most important features to measure customer engagement which is missing in csv.
