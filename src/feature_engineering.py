import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer

def add_pre_churn_features(data: DataFrame) -> DataFrame:
    """
    Adds pre-churn features to the data.
    
    Parameters:
    data (DataFrame): The input data with columns 'gaming_turnover_sum', 'betting_turnover_sum', 'player_key', and 'date'.
    
    Returns:
    DataFrame: The data with additional pre-churn features.
    """
    data['total_turnover'] = data['gaming_turnover_sum'] + data['betting_turnover_sum']  # Combine turnover
    data['active'] = np.where(data['total_turnover'] > 0, 1, 0)  # Create 'active' column
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


def add_churn_as_target(data: DataFrame, inactivity_threshold: int = 30) -> DataFrame:
    """
    Adds churn as a target feature to the data.
    
    Parameters:
    data (DataFrame): The input data with columns 'player_key' and 'date'.
    inactivity_threshold (int): Number of days for inactivity checks (default is 30 days).
    
    Returns:
    DataFrame: The data with churn target feature.
    """
    data = data.sort_values(by=['player_key', 'date']).copy()
    data['churn'] = 0  # Initialize churn column

    for player_key, group in data.groupby('player_key'):
        group = group.set_index('date').resample('D').asfreq().fillna(method='ffill').reset_index()
        last_bet_date = None
        for i in range(len(group)):
            if group['active'].iloc[i] == 1:
                last_bet_date = group['date'].iloc[i]
                data.loc[group.index[i:], 'churn'] = 0  # Reset churn status if the customer places a bet
            elif last_bet_date is not None:
                days_since_last_bet = (group['date'].iloc[i] - last_bet_date).days
                if days_since_last_bet > inactivity_threshold:
                    data.loc[group.index[i:], 'churn'] = 1  # Mark churn for all subsequent rows
                    break

    return data

def create_time_series_data(df, time_window='W'):  # Default to weekly windows
    """
    Calculates time series features for the data and marks churn if the user goes missing.
    
    Parameters:
    df (DataFrame): The input data with columns 'player_key', 'date', and other relevant features.
    inactivity_threshold (int): Number of days for inactivity checks (default is 30 days).
    
    Returns:
    DataFrame: The data with additional time series features and churn marked.
    """

    df = df.sort_values(['player_key', 'date']).copy()  # Ensure proper order
    df['date'] = pd.to_datetime(df['date']) # Convert to datetime objects.

    df = df.set_index('date') # Set date as index to use resample function.

    time_series_data = []
    for player_key, group in df.groupby('player_key'):
        resampled_group = group.resample(time_window).agg({
            'total_turnover': 'sum',
            'gaming_turnover_sum': 'sum',
            'betting_turnover_sum': 'sum',
            'deposit_sum': 'sum',
            'withdrawal_sum': 'sum',
            'login_num': 'sum',
            'active': 'sum', # Number of active days in time window.
            'inactive_days': 'max', # Max inactive days in time window.
            'churn': 'max', # Churn status for the time window.
            'betting_NGR': 'sum',
            'gaming_NGR': 'sum',
            'net_deposits': 'sum'
        }).fillna(0) # Fill NaN with 0.

        resampled_group['player_key'] = player_key
        time_series_data.append(resampled_group.reset_index()) # Add index as a column.

    final_df = pd.concat(time_series_data)  
    final_df['rolling_mean_turnover_4w'] = final_df.groupby('player_key')['total_turnover'].rolling(window=4).mean().reset_index(level=0, drop=True).fillna(0)
    final_df['turnover_change'] = final_df.groupby('player_key')['total_turnover'].pct_change().fillna(0)  # Fill NaN with 0.
    final_df = final_df.fillna(0)  # Fill NaN with 0 after feature engineering.
    return final_df



def calculate_time_series_features(df: DataFrame, inactivity_threshold: int = 30) -> DataFrame:
    """
    Calculates time series features for the data and marks churn if the user goes missing.
    
    Parameters:
    df (DataFrame): The input data with columns 'player_key', 'date', and other relevant features.
    inactivity_threshold (int): Number of days for inactivity checks (default is 30 days).
    
    Returns:
    DataFrame: The data with additional time series features and churn marked.
    """
    time_series_data = []
    df = df.sort_values(['player_key', 'date']).copy()  # Ensure proper order
    df = df.set_index('date') # Set date as index to use resample function.

    time_series_data = []
    for player_key, group in df.groupby('player_key'):
        resampled_group = group.resample('W').agg({
            'total_turnover': 'sum',
            'gaming_turnover_sum': 'sum',
            'betting_turnover_sum': 'sum',
            'deposit_sum': 'sum',
            'withdrawal_sum': 'sum',
            'login_num': 'sum',
            'active': 'sum', # Number of active days in time window.
            'inactive_days': 'max', # Max inactive days in time window.
            'churn': 'max', # Churn status for the time window.
            'betting_NGR': 'sum',
            'gaming_NGR': 'sum',
            'net_deposits': 'sum'
        }).fillna(0) # Fill NaN with 0.

        resampled_group['player_key'] = player_key
        time_series_data.append(resampled_group.reset_index()) # Add index as a column.

    final_df = pd.concat(time_series_data)  
    final_df['rolling_mean_turnover_4w'] = final_df.groupby('player_key')['total_turnover'].rolling(window=4).mean().reset_index(level=0, drop=True).fillna(0)
    final_df['turnover_change'] = final_df.groupby('player_key')['total_turnover'].pct_change().fillna(0)  # Fill NaN with 0.
    final_df = final_df.fillna(0)  # Fill NaN with 0 after feature engineering.
    return final_df

def clean_transaction_frame(X: DataFrame) -> DataFrame:
    """
    Cleans the transaction data by imputing missing values.
    
    Parameters:
    X (DataFrame): The input data.
    
    Returns:
    DataFrame: The cleaned data.
    """
    imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent', etc.
    X_imputed = imputer.fit_transform(X)  # Fit and transform X
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)  # Convert back to DataFrame
    return X
def clean_timeseries_frame(df: DataFrame, fill_val: float = 0, clip_lower: float = -1e5, clip_upper: float = 1e5) -> DataFrame:
    """
    Cleans the time series frame by performing the following steps:
    1) Replace ±∞ with NaN
    2) Fill NaN with a specified value (default=0)
    3) Clip extreme values to a specified range [clip_lower, clip_upper]
    
    Parameters:
    df (DataFrame): The input data frame to be cleaned.
    fill_val (float): The value to replace NaNs with (default is 0).
    clip_lower (float): The lower bound to clip the values (default is -1e5).
    clip_upper (float): The upper bound to clip the values (default is 1e5).
    
    Returns:
    DataFrame: The cleaned data frame.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(fill_val)
    df = df.clip(lower=clip_lower, upper=clip_upper)
    return df

def get_raw_transaction_features(df: DataFrame) -> DataFrame:
    """
     Extracts new features to enhance churn prediction for raw transaction data

    1. Bet frequency ratio: total bet count / total login count.
    2. Periodic net deposit difference
    3. Turnover-volatility ratio: standard deviation of turnover / mean turnover.
    """
    # Total turnover
    df['total_turnover'] = df['gaming_turnover_sum'] + df['betting_turnover_sum']

    # Bet frequency ratio: bet count / login count
    # Replace zero login counts to avoid inf
    df['login_num'] = df['login_num'].replace(0, np.nan)
    df['bet_frequency_ratio'] = (df['gaming_turnover_num'] + df['betting_turnover_num']) / df['login_num']
    df['bet_frequency_ratio'] = df['bet_frequency_ratio'].fillna(0)

    # Turnover volatility ratio (if you have multiple daily records per player):
    # For demonstration, use stdev of gaming turnover num + betting turnover num
    df['turnover_count_std'] = (df[['gaming_turnover_num', 'betting_turnover_num']]
                                 .std(axis=1))
    df['turnover_count_mean'] = (df[['gaming_turnover_num', 'betting_turnover_num']]
                                  .mean(axis=1))
    df['turnover_volatility_ratio'] = df.apply(
        lambda row: row['turnover_count_std'] / row['turnover_count_mean']
        if row['turnover_count_mean'] != 0 else 0, axis=1
    )

    # Periodic net deposit difference
    df['net_deposits'] = df['deposit_sum'] - df['withdrawal_sum']
    return df

def define_churn(data: DataFrame, inactivity_threshold: int = 30) -> DataFrame:
    """
    Defines churn based on inactivity for a specified threshold of days.
    
    Parameters:
    data (DataFrame): The input data with columns 'player_key', 'date', and 'total_turnover'.
    inactivity_threshold (int): Number of days for inactivity checks (default is 30 days).
    
    Returns:
    DataFrame: The data with churn defined.
    """
    """
    data: DataFrame with columns
    lookback_days: number of days for inactivity checks
    """
    # Ensure data is sorted for group-based rolling windows
    data = data.sort_values(by=['player_key', 'date']).copy()

    # Calculate daily net deposits (deposits minus withdrawals)
    data['net_deposits'] = data['deposit_sum'] - data['withdrawal_sum']
    
    # Calculate daily total turnover
    data['total_turnover'] = data['gaming_turnover_sum'] + data['betting_turnover_sum']

    # Rolling sums need a fixed frequency, so group by player and use rolling on sorted dates.
    def rolling_features(group):
        group = group.set_index('date')
        # n-day rolling sums for bets
        group['bets_nd'] = group['total_turnover'].rolling(f'{inactivity_threshold}D').sum()
        return group.reset_index()

    data = data.groupby('player_key', group_keys=False).apply(rolling_features)

    # Now define churn flags for each criterion:
    # 1) No bets placed in the last n days
    data['flag_no_bets'] = data['bets_nd'] == 0
    # 2) Identify each player's earliest date using a ranking approach
    data['is_first_date'] = data.groupby('player_key')['date'].rank(method='first') == 1
    # 3) Force the first date for each player to be False for 'flag_no_bets'
    data.loc[data['is_first_date'], 'flag_no_bets'] = False
    data['churn'] = data['flag_no_bets']       
    return data