# Import all necessary modules 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def load_data(path):
    # Import Dataset 
    df = pd.DataFrame(pd.read_csv(path))
    print("imported data...")
    return df

def preprocessing(data):
    # Convert category_data into numeric_data for computation
    # It's Binary encoding technique which is only 0 & 1
    data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})
    data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'No':0, 'Yes':1})

    # One-hot-Encoding 
    data = pd.get_dummies(data, columns=['Region_Code', 'Vehicle_Age', 'Policy_Sales_Channel'], drop_first=True)
    print("converted all categorical columns into numeric columns....")

    return data 

def train_model(df):
    # split features and labels 
    X = df.drop(columns=['id', 'Response'], axis=1)
    y = df['Response']

    # Split Data into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Train test split done....")

    # fix all outlier by capping 
    Q1 = X_train['Annual_Premium'].quantile(0.25)
    Q3 = X_train['Annual_Premium'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    X_train['Annual_Premium'] = np.where(X_train['Annual_Premium'] < lower_bound, lower_bound, np.where(X_train['Annual_Premium'] > upper_bound, upper_bound, X_train['Annual_Premium']))

    # Apply the X_train parameters to the X_test 
    X_test['Annual_Premium'] = np.where(X_test['Annual_Premium'] < lower_bound, lower_bound, np.where(X_test['Annual_Premium'] > upper_bound, upper_bound, X_test['Annual_Premium']))
    print("Applied all parameters from train to test....")

    # selected model(Random Forest) amoung all models 
    # Tunning hyper-parameters for 
    param_grid = {
        'n_estimators' : [100, 200],
        'max_depth' : [5, 10],
        'min_samples_leaf' : [1, 3]
    }

    print("Model training....")
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    model = RandomForestClassifier(random_state=42, **best_params, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training done....")

    # downloading train data 
    dump(X.columns.tolist(), r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\model_columns.joblib")
    dump((lower_bound, upper_bound), r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\premium_bounds.joblib")
    dump(model, r"C:\Users\hp5cd\Documents\machine learning\capstone project\model file\RF_model.joblib")
    print("Saved in your directory")

def main():
    # Provide the correct file path to the location where the dataset is stored 
    path = r"C:\Users\hp5cd\Documents\machine learning\capstone project\Data\data.csv"
    df = load_data(path)
    data = preprocessing(df)
    train_model(data)

main()
