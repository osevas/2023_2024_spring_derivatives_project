"""
Object that fetches dataset from database
Author: Onur Sevket Aslan
Date: 2024-05-01
"""
import pandas as pd
import yfinance as yf

class DatasetOpener:
    """
    Object that fetches dataset from dataset file
    """
    def __init__(self, stock_name):
        """
        Initializing by using stock anme

        Args:
            stock_name (str): Stock name of the stock (for example: AAL)
        """
        self.stock_name = stock_name
        stock_dict = {'AAL': 'American Airlines', 'GC=F': 'Gold_Futures'}
        print('Opening ' + self.stock_name + ': '+ stock_dict[self.stock_name])
    
    def import_txt(self):
        """
        Function that will open txt file and retrieve data as pandas dataframe

        Args:
            stock_name (string): stock code
        """
        dataframe = pd.read_csv('./datasets/' + self.stock_name + '.txt')
        return dataframe
    
    def import_from_yfinance(self):
        """
        Function that will download data from yfinance
        """
        msft = yf.Ticker(self.stock_name)
        # print(msft.info)
        data_gold_ft = msft.history(period="max")
        # print(type(data_gold_ft))

        return data_gold_ft

        