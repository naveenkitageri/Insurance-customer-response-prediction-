# Insurance-customer-response-prediction-
Bulit an end-to-end Machine Learning pipeline to predict customer response to insurance policy offers using multiple classification models, including data preprocessing, feature engineering and hyperparameter tuning.

- This project predicts whether a customer will respond (yes/no) to an insurance offer using various Machine Learning algorithm and selecting the best -performing model.

# Project objective
- To bulit a classification model that accurately predict customer response based on:
  - Customer demographics
  -   Policy-related information
  - Financial attributes
  - Marketing interaction features

# Dataset 
- Source:Insurance Customer Dataset

  Target Variable:
  **Response(1=Yes, 0=No)**  

  Feature variable:
  - ID
  - Age
  - Gender
  - Annual Premium
  - Driving License   
  - Region Code
  - Vehicle Age
  - Previously Insured
  - Vehicle Damage
  - Policy Sales Channel
  - Vintage
 
# WorkFlow 
  1. Data loading
  2. Exploratory data analysis (EDA)
  3. Outlier detection and treatment
  4. Feature engineering
  5. Train test split
  6. Feature scaling
  7. Model training
  8. Model evaluation
  9. Cross validation
  10. Hyperparameter tuning (GridSearchCV)
  11. Best model selection
  12. Model saving

# Model performance model :  Accuracy score, recall
1.Linear Regression : 0.88, 50 |
2.Decision Tree : 0.83, 60 |
3.Tuned Decision Tree : 0.70, 80 |
4.Random Forest : 0.86, 55 | 
5.Tuned Random Forest : 0.71, 79 |

# Final Model 
- Selected Model : **Tuned Random Forest**
- Reason : Best overall performance with optimized hyperparameters
- Saved Using : Joblib


# Result and Business Impact 
- Helps insurance companies identify high potential customers
- Improve marketing campaign efficiency
- Reduces operational costs
- Enhances customer targeting strategy
