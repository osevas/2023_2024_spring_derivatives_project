"""
Objective: Project of FE604 Advanced Derivatives
Author: https://github.com/osevas
Date: 2024-05-01
Reference: https://https://github.com/osevas/2023_2024_spring_derivatives_project
"""
from dataset_opener import DatasetOpener
# from data_cleaner import DataCleaner
# from data_visual import DataVisual
# from data_analyzer import DataAnalyzer

def open_dataset():
    dataset_opener_1 = DatasetOpener('GC=F') # Gold Futures
    data_gold_ft = dataset_opener_1.import_from_yfinance() # download data from yfinance as pandas dataframe

def main():
    # open dataset
    open_dataset()

if __name__ == "__main__":
    main()