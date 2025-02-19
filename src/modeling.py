import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_path: str = 'churn_model.json'):
    """
    Standardizes the features, trains the model, makes predictions, evaluates the model, and saves the model.
    
    Parameters:
    X_train (DataFrame): The training features.
    X_test (DataFrame): The test features.
    y_train (Series): The training labels.
    y_test (Series): The test labels.
    model_path (str): The path to save the trained model (default is 'churn_model.json').
    
    Returns:
    None
    """

    # Train the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    model.save_model(model_path)