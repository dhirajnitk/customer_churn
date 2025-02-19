# The main.py script is the entry point for running the churn prediction model. It takes two arguments: --approach and --enable-eda. The --approach argument specifies the approach to use for the churn prediction model, which can be either 'transaction' or 'time_series'. The --enable-eda argument enables exploratory data analysis (EDA) and plotting if specified.
# To run the file from the command line, use the following command:
#python main.py --approach transaction'
#python main.py --approach time_series
# you can enable EDA and plotting by adding the --enable-eda but please dont run this

import argparse
import pandas as pd
import os
from src.data_processing import process_data
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.feature_engineering import clean_transaction_frame, clean_timeseries_frame, clean_transaction_frame, create_time_series_data, define_churn, calculate_time_series_features, get_raw_transaction_features
from src.sampling import resample_data
from src.modeling import train_and_evaluate_model
from src.plotting import plot_all_players_same_plot
from src.eda import run_eda

def process_transaction_data(file_path: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Processes the transaction data by loading, preparing, and calculating necessary features.
    
    Parameters:
    file_path (str): The path to the CSV file.
    lookback_days (int): The number of days to look back for churn calculation (default is 30 days).
    
    Returns:
    DataFrame: The processed data.
    """
    data = process_data(file_path,lookback_days=lookback_days )
    data = get_raw_transaction_features(data)
    return data

def process_time_series_data(file_path: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Processes the time series data by loading, preparing, and calculating necessary features.
    
    Parameters:
    file_path (str): The path to the CSV file.
    lookback_days (int): The number of days for inactivity checks (default is 30 days).
    
    Returns:
    DataFrame: The processed data.
    """
    data = process_data(file_path,lookback_days=lookback_days)
    data = calculate_time_series_features(data)
    return data

def main():
    parser = argparse.ArgumentParser(description='Run churn prediction model.')
    parser.add_argument('--approach', type=str, choices=['transaction', 'time_series'], required=True, help='Choose the approach: transaction or time_series')
    parser.add_argument('--enable-eda', action='store_true', help='Enable EDA and plotting')
    args = parser.parse_args()

    # File path to the data
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    
    # Load data for EDA
    data = process_data(file_path)
    if args.enable_eda:
        # Perform EDA 
        run_eda(data)
        # Plot all players on the same plot for total turnover
        plot_all_players_same_plot(data, column_to_plot='total_turnover', number_inactive_days=30, no_of_players=10)

    if args.approach == 'transaction':
        # Process transaction data
        transaction_data = process_transaction_data(file_path, lookback_days=7)
        
        # Define features and target for transaction data
        transaction_features = [
            'birth_year', 'gaming_turnover_sum', 'gaming_turnover_num', 'gaming_NGR',
            'betting_turnover_sum', 'betting_turnover_num', 'betting_NGR',
            'deposit_sum', 'deposit_num', 'withdrawal_sum', 'withdrawal_num', 'login_num',
            'total_turnover', 'bet_frequency_ratio', 'turnover_count_std', 'turnover_count_mean',
            'turnover_volatility_ratio', 'net_deposits'
        ]
        transaction_target = 'churn'
          # Clean the data
        X = transaction_data[transaction_features]
        X_cleaned = clean_transaction_frame(X)
        transaction_data[transaction_features] = X_cleaned
        # Resample transaction data
        X_train, X_test, y_train, y_test = resample_data(transaction_data, transaction_features, transaction_target, enable_new_features=False, test_size=0.3, random_state=42)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_model_transaction.json')
           # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate model on transaction data
        train_and_evaluate_model(X_train, X_test, y_train, y_test, model_path=model_path)

    
    elif args.approach == 'time_series':
        # Process time series data
        time_series_data = process_time_series_data(file_path, inactivity_threshold=30)
        
        # Define features and target for time series data
        time_series_features = [
            'total_turnover', 'gaming_turnover_sum', 'betting_turnover_sum', 'deposit_sum',
            'withdrawal_sum', 'login_num', 'rolling_mean_turnover_4w', 'turnover_change',
            'betting_NGR', 'gaming_NGR', 'net_deposits'
        ]
        time_series_target = 'churn'
        
          # Clean the data
        X = time_series_data[time_series_features]
        y = time_series_data['churn']
        X_cleaned = clean_timeseries_frame(X)
        time_series_data[time_series_features] = X_cleaned

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_cleaned)
        # Time-based Data Splitting (Crucial!)
        train_cutoff = time_series_data['date'].quantile(0.7)  # 70% for training
        X_train = X_scaled[time_series_data['date'] <= train_cutoff]
        y_train = y[time_series_data['date'] <= train_cutoff]
        X_test = X_scaled[time_series_data['date'] > train_cutoff]
        y_test = y[time_series_data['date'] > train_cutoff]
                # Resample time series data
      
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_model_time_series.json')
        # Train and evaluate model on time series data
        train_and_evaluate_model(X_train, X_test, y_train, y_test, model_path=model_path)
        


if __name__ == "__main__":
    main()



