import pandas as pd
import numpy as np
from src.feature_engineering import define_churn

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Loads and prepares the data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: The prepared data.
    """
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['player_key', 'date'])
    return data

def calculate_total_turnover(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the total turnover by combining gaming and betting turnover.
    
    Parameters:
    data (DataFrame): The input data.
    
    Returns:
    DataFrame: The data with the total turnover calculated.
    """
    data['total_turnover'] = data['gaming_turnover_sum'] + data['betting_turnover_sum']
    return data

def create_active_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an 'active' column: 1 if active (any turnover), 0 if inactive.
    
    Parameters:
    data (DataFrame): The input data.
    
    Returns:
    DataFrame: The data with the 'active' column added.
    """
    data['active'] = np.where(data['total_turnover'] > 0, 1, 0)
    return data

def calculate_inactive_days(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of inactive days for each player.
    
    Parameters:
    data (DataFrame): The input data.
    
    Returns:
    DataFrame: The data with the 'inactive_days' column added.
    """
    data['inactive_days'] = 0.0  # Initialize as float to handle NaNs

    for player_key, group in data.groupby('player_key'):
        last_active_date = None
        for i in range(len(group)):
            if group['active'].iloc[i] == 1:
                last_active_date = group['date'].iloc[i]
                data.loc[group.index[i], 'inactive_days'] = 0  # Reset inactive days
            elif last_active_date is not None:  # Only calculate if there was previous activity
                inactive_days = (group['date'].iloc[i] - last_active_date).days
                data.loc[group.index[i], 'inactive_days'] = inactive_days

    return data

def process_data(file_path: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Processes the data by loading, preparing, and calculating necessary features.
    
    Parameters:
    file_path (str): The path to the CSV file.
    lookback_days (int): The number of days to look back for churn calculation (default is 30 days).
    
    Returns:
    DataFrame: The processed data.
    """
    data = load_and_prepare_data(file_path)
    data = calculate_total_turnover(data)
    data = create_active_column(data)
    data = calculate_inactive_days(data)
    data = define_churn(data, inactivity_threshold=lookback_days)
    return data

def print_player_data(data: pd.DataFrame):
    """
    Prints relevant columns for each player.
    
    Parameters:
    data (DataFrame): The input data.
    """
    for player_key, group in data.groupby('player_key'):
        print(f"Player: {player_key}")
        print(group[['date', 'net_deposits', 'total_turnover', 'active', 'inactive_days', 'churn']])  # Print relevant columns
        print("-" * 20)