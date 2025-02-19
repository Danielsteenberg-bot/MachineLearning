import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def simplePlot(data):
    # Opret figure og axis objekter
    fig, ax = plt.subplots()
    
    # Liste over kolonner vi ikke vil plotte
    exclude_columns = ["month_number", "total_units", "total_profit"]
    
    # Plot hver kolonne der ikke er i exclude_columns
    for column in data.columns:
        if column not in exclude_columns:
            ax.plot(data["month_number"], data[column], label=column)
    
    # Sæt labels og titel
    ax.set(xlabel='Måned', ylabel='Produkt', title='Salg over tid')
    ax.legend()
    
    return plt.show()
def createPieChart(data):
    """
    Opret et pie chart der viser fordelingen af produktsalg
    
    Args:
        data (pd.DataFrame): DataFrame med salgsdata
    """
    # Opret figure og axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Kolonner vi ikke vil have med i pie chart
    exclude_columns = ["month_number", "total_units", "total_profit"]
    
    # Beregn total salg for hvert produkt
    product_sales = {}
    for column in data.columns:
        if column not in exclude_columns:
            product_sales[column] = data[column].sum()
    
    # Forbered data til pie chart
    labels = list(product_sales.keys())
    sizes = list(product_sales.values())
    
    # Opret pie chart
    ax.pie(sizes, 
           labels=labels, 
           autopct='%1.1f%%',  # Vis procenter med 1 decimal
           startangle=90)      # Start ved 90 grader
    
    ax.set_title('Fordeling af totalt salg per produkt')
    
    # Sørg for at cirklen er rund
    plt.axis('equal')
    
    return plt.show()

def createStackPlot(data):
    """
    Opret et stacked plot der viser akkumuleret salg af produkter over tid
    
    Args:
        data (pd.DataFrame): DataFrame med salgsdata
    """
    # Opret figure og axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Kolonner vi ikke vil have med i stack plot
    exclude_columns = ["month_number", "total_units", "total_profit"]
    
    # Forbered data til stack plot
    x = data["month_number"]
    ys = [data[column] for column in data.columns if column not in exclude_columns]
    labels = [column for column in data.columns if column not in exclude_columns]
    
    # Opret stack plot
    ax.stackplot(x, ys, labels=labels)
    
    # Tilføj labels og titel
    ax.set(xlabel='Måned', 
           ylabel='Akkumuleret salg', 
           title='Akkumuleret produktsalg over tid')
    
    # Tilføj legend i top-right hjørne
    ax.legend(loc='upper right')
    
    # Tilføj grid for bedre læsbarhed
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return plt.show()