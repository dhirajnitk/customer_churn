import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

def plot_all_players_same_plot(df: pd.DataFrame, column_to_plot: str, number_inactive_days: int = 30, no_of_players: int = 10):
    """
    Plots all players on the same plot for a given column, with different line styles for continuous, discontinuous, and non-continuous data.
    
    Args:
        df (DataFrame): The preprocessed DataFrame.
        column_to_plot (str): The name of the column to plot.
        number_inactive_days (int): The threshold in days to determine discontinuity (default is 30 days).
        no_of_players (int): The number of players to plot (default is 10).
    """
    plt.figure(figsize=(15, 7))  # Adjust figure size as needed

    players = df['player_key'].unique()[:no_of_players]
    colors = cm.get_cmap('tab10', len(players))  # Get a colormap with a unique color for each player

    for idx, player in enumerate(players):
        player_data = df[df['player_key'] == player].sort_values(by='date').reset_index(drop=True)
        if not player_data.empty:  # Check if player has data for the column
            # Calculate the difference in days between consecutive dates
            player_data['date_diff'] = player_data['date'].diff().dt.days
            plt.figure(figsize=(15, 7))  # Adjust figure size as needed

    players = df['player_key'].unique()[:no_of_players]
    colors = cm.get_cmap('tab10', len(players))  # Get a colormap with a unique color for each player

    for idx, player in enumerate(players):
        player_data = df[df['player_key'] == player].sort_values(by='date').reset_index(drop=True)
        if not player_data.empty:  # Check if player has data for the column
            plt.plot(player_data['date'], player_data[column_to_plot], linestyle='-', color=colors(idx), label=f"Player {player}")

    plt.title(f"{column_to_plot} Over Time for {no_of_players} Players")
    plt.xlabel("Date")
    plt.ylabel(column_to_plot)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(loc='best', fontsize='small')  # Adjust fontsize as needed
    plt.tight_layout()
    plt.show()

