import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def prepare_tournament_data(csv_path):
    """
    Prepares tournament data for analysis by cleaning and structuring the data.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Fill empty values with 0
    df = df.fillna(0)
    df = df.replace('', 0)
    
    # Convert all values to numeric
    for col in df.columns:
        if col != 'Game':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def get_contrasting_colors(n):
    """Generate n distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0
        value = 0.8 + 0.2 * (i % 2)  # Alternate between 0.8 and 1.0
        colors.append(plt.cm.hsv(hue))
    return colors

def plot_cumulative_scores(df):
    """Plot cumulative scores for all players with minimal white space"""
    # Create a copy of the dataframe excluding the 'Game' column
    score_df = df.drop('Game', axis=1)
    
    # Calculate cumulative sums
    cumulative_scores = score_df.cumsum()
    
    # Create figure with reduced size
    fig = plt.figure(figsize=(14, 8))
    
    # Create main plot with compact positioning
    ax1 = plt.axes([0.08, 0.22, 0.9, 0.73])  # [left, bottom, width, height]
    
    # Get contrasting colors
    colors = get_contrasting_colors(len(score_df.columns))
    
    # Dictionary to store final scores for label positioning
    final_scores = defaultdict(list)
    
    # Plot each player's cumulative score with varying line styles
    line_styles = ['-', '--', '-.', ':']
    lines = []
    for idx, column in enumerate(cumulative_scores.columns):
        line_style = line_styles[idx % len(line_styles)]
        final_score = cumulative_scores[column].iloc[-1]
        
        line = ax1.plot(df['Game'], cumulative_scores[column],
                       label=column,
                       color=colors[idx],
                       linewidth=2,
                       linestyle=line_style,
                       alpha=0.8,)[0]
        lines.append(line)
        
        final_scores[final_score].append(column)
    
    # offset labels at the end of lines
    max_game = df['Game'].max()
    y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    offset = y_range * 0.01  # 1% of the y-range
    
    # Sort final scores for better label placement
    for final_score in sorted(final_scores.keys()):
        players = final_scores[final_score]
        if len(players) > 1:
            label = ' & '.join(players)
        else:
            label = players[0]
            
        ax1.annotate(label,
                    xy=(max_game, final_score),
                    xytext=(max_game + 0.5, final_score),  # Reduced offset
                    verticalalignment='center',
                    fontsize=8)
    
    ax1.set_title('Quing of the South 2024 - Scorecard', fontsize=14, pad=10)
    ax1.set_xlabel('Game #', fontsize=10, labelpad=5)
    ax1.set_ylabel('Cumulative Points', fontsize=10, labelpad=5)
    
    # Add grid with lower opacity
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Set minimal margins
    ax1.margins(x=0.01, y=0.02)
    
    # Adjust x-axis limits
    ax1.set_xlim(-0.5, max_game + 12)  # padding for annotations
    
    # add legend
    ncols = min(8, len(lines))
    legend = ax1.legend(lines, [line.get_label() for line in lines],
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.12), 
                       ncol=ncols,
                       fontsize=9,
                       frameon=True,
                       borderaxespad=0.,
                       handlelength=1.5,
                       columnspacing=1.0)
    
    # Adjust spacing between subplot elements - minimal margins
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.98)
    
    return fig

if __name__ == "__main__":
    
    csv_path = "raw_data/quing_of_the_south_2024_results.csv"
    
    df = prepare_tournament_data(csv_path)
    
    # Create and save the cumulative scores plot
    fig = plot_cumulative_scores(df)
    plt.savefig('quing_of_the_south_2024_results.png', dpi=300, bbox_inches='tight')