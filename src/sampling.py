import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def resample_data(data: pd.DataFrame, features: list, target: str, enable_new_features: bool = False, test_size: float = 0.3, random_state: int = 42):
    """
    Undersamples the majority class and oversamples the minority class.
    
    Parameters:
    data (DataFrame): The input data.
    features (list): The list of feature columns.
    target (str): The target column.
    enable_new_features (bool): Flag to enable new features (default is False).
    test_size (float): The proportion of the dataset to include in the test split (default is 0.3).
    random_state (int): The random state for reproducibility (default is 42).
    
    Returns:
    tuple: The resampled training and test sets (X_train, X_test, y_train, y_test).
    """
    X = data[features]
    y = data[target]

    if enable_new_features:
        # Strategy 1: Imputation (replace NaNs with a value)
        imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent', etc.
        X_imputed = imputer.fit_transform(X)  # Fit and transform X
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)  # Convert back to dataframe

    # Oversample with SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy=0.5)  # Example: Oversample to 50% of majority class
    X_over, y_over = smote.fit_resample(X, y)

    # Undersample
    rus = RandomUnderSampler(random_state=random_state, sampling_strategy='majority')  # Example: Undersample majority to be equal to minority
    X_resampled, y_resampled = rus.fit_resample(X_over, y_over)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

