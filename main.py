import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def prepare_tournament_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare tournament data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing tournament data
        
    Returns:
        DataFrame with cleaned numeric data
    """
    df = pd.read_csv(csv_path)
    df = df.fillna(0).replace('', 0)
    numeric_cols = df.columns.difference(['Game'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def generate_plot_styles(n: int) -> Tuple[List[tuple], List[str]]:
    """
    Generate distinct colors and line styles for plotting.
    
    Args:
        n: Number of styles needed
        
    Returns:
        Tuple of (colors, line_styles)
    """
    colors = [plt.cm.hsv(i/n) for i in range(n)]
    line_styles = ['-', '--', '-.', ':']
    return colors, line_styles

def plot_cumulative_scores(df: pd.DataFrame, figsize: tuple) -> plt.Figure:
    """
    Create a cumulative scores plot for tournament players.
    
    Args:
        df: Tournament data DataFrame with 'Game' column and player scores
        
    Returns:
        Matplotlib figure object
    """
    score_df = df.drop('Game', axis=1)
    cumulative_scores = score_df.cumsum()
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0.08, 0.22, 0.9, 0.73])
    
    colors, line_styles = generate_plot_styles(len(score_df.columns))
    final_scores: Dict[float, List[str]] = defaultdict(list)
    lines = []
    
    # Plot player scores
    for idx, column in enumerate(cumulative_scores.columns):
        line = ax.plot(
            df['Game'],
            cumulative_scores[column],
            label=column,
            color=colors[idx],
            linewidth=2,
            linestyle=line_styles[idx % len(line_styles)],
            alpha=0.8
        )[0]
        lines.append(line)
        final_scores[cumulative_scores[column].iloc[-1]].append(column)
    
    # Add end-of-line labels
    max_game = df['Game'].max()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    for final_score, players in sorted(final_scores.items()):
        label = ' & '.join(players)
        ax.annotate(
            label,
            xy=(max_game, final_score),
            xytext=(max_game + 0.5, final_score),
            verticalalignment='center',
            fontsize=8
        )
    
    # Configure plot styling
    ax.set_title('Quing of the South 2024 - Scorecard', fontsize=14, pad=10)
    ax.set_xlabel('Game #', fontsize=10, labelpad=5)
    ax.set_ylabel('Cumulative Points', fontsize=10, labelpad=5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.margins(x=0.01, y=0.02)
    if figsize[0] == figsize[1]:
        added_margin = 24
    else:
        added_margin = 12
    ax.set_xlim(-0.5, max_game + added_margin)
    
    # Configure legend
    ncols = min(8, len(lines))
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncols,
        fontsize=9,
        frameon=True,
        borderaxespad=0.,
        handlelength=1.5,
        columnspacing=1.0
    )
    
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.98)
    return fig

if __name__ == "__main__":
    CSV_PATH = "raw_data/quing_of_the_south_2024_results.csv"
    tournament_data = prepare_tournament_data(CSV_PATH)
    
    # Wide Image
    figure = plot_cumulative_scores(tournament_data, figsize=(14, 8))
    figure.savefig('quing_of_the_south_2024_results.png', dpi=300, bbox_inches='tight')
    
    # Square Image
    figure = plot_cumulative_scores(tournament_data, figsize=(8, 8))
    figure.savefig('quing_of_the_south_2024_results_square.png', dpi=300, bbox_inches='tight')