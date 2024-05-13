"""
Objective: Project of FE604 Advanced Derivatives
Author: https://github.com/osevas
Date: 2024-05-01
Reference: https://https://github.com/osevas/2023_2024_spring_derivatives_project
Environment activation: 
C:\\Users\\oseva\\Documents\\py_envs\\deriv_env\\Scripts\\activate
"""

import os
from dataset_opener import DatasetOpener
# from data_cleaner import DataCleaner
from data_visual import DataVisual
from data_analyzer import DataAnalyzer
from monte_carlo import MonteCarlo

def delete_png_files():
    """
    Function that deletes png files
    """
    for file in os.listdir():
        if file.endswith('.png'):
            os.remove(file)
    return None

def open_dataset():
    """
    Function that opens dataset

    Returns:
        _type_: pandas dataframe
    """
    dataset_opener_1 = DatasetOpener('GC=F') # Gold Futures
    data_gold_ft = dataset_opener_1.import_from_yfinance() # download data from yfinance as pandas dataframe
    return data_gold_ft

def analyze_dataset(data):
    """
    initial analysis of dataset

    Args:
        data_gold_ft (_type_): _description_
    """
    data_analyzer_1 = DataAnalyzer(data, 'GC=F')
    data_analyzer_1.calculate_daily_return('Close')
    data_analyzer_1.print_summary()
    return None

def visualize_dataset(data):
    """
    initial visualization of dataset

    Args:
        data_gold_ft (_type_): _description_
    """
    data_visual_1 = DataVisual()
    data_visual_1.plt_time_series(data, 'GC=F', 'High')
    data_visual_1.plt_time_series(data, 'GC=F', 'daily_return')
    return None



def main():
    """
    main function
    """
    # delete png files
    delete_png_files()

    # open dataset
    data_main = open_dataset()

    # initial analysis of dataset
    # analyze_dataset(data_main)

    # initial visualization of dataset
    # visualize_dataset(data_main)

    # monte carlo simulation
    monte_carlo_1 = MonteCarlo(data_main, 'GC=F', 1260, 5, 'Close', 100)
    monte_carlo_1.simulate()


if __name__ == "__main__":
    main()