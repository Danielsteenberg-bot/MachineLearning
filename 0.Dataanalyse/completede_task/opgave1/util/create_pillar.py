import matplotlib.pyplot as plt
import pandas as pd

def create_pillar_diagram(data, title='', xlabel='', ylabel='', stacked=False):
    """
    Create a simple pillar (bar) diagram with sensible defaults.
    
    Args:
        data: Dictionary of pandas Series to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        stacked: If True, creates stacked bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if stacked:
        df = pd.DataFrame(data)
        df.plot(kind='bar', stacked=True, ax=ax)
    else:
        # Handle single series
        for name, series in data.items():
            ax.bar(series.index, series.values, label=name)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    if not stacked:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f')
    
    # Add legend if multiple series
    if len(data) > 1 or stacked:
        ax.legend(title='Kategorier')
    
    plt.tight_layout()
    plt.show()