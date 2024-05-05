"""
Objective: to create an object for analyzing
Author: Onur S. Aslan
Date: 2024-05-02
"""
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from data_visual import DataVisual

class DataAnalyzer:
    """
    to create an object for analyzing
    """
    def __init__(self, dataframe, stock_symbol):
        """
        Initialization of dataAnalyzer.  

        Args:
            dataframe (pandas dataframe): stock data (output of DataCleaner)
        """
        print('\nAnalyzing stock ' + stock_symbol + '...\n')
        self.data = dataframe
        self.stock_symbol = stock_symbol
    
    def print_head_tail(self):
        """
        printing head and tail of dataframe
        """
        print("Head of dataframe: \n")
        print(self.data.head())
        print("\nTail of dataframe: \n")
        print(self.data.tail())
        return None
    
    def print_describe(self):
        """
        printing describe of dataframe
        """
        print('\nPrinting summary of stock data:\n')
        print(self.data.describe())
        return None\
    
    def print_info(self):
        """
        printing info of dataframe
        """
        print('\nPrinting info of stock data:\n')
        print(self.data.info())
        return None

    def print_summary(self):
        """
        printing summary of dataframe

        Args:
            df (pandas dataframe): _description_
        """
        self.print_head_tail()
        self.print_describe()
        self.print_info()
        return None
    
    def decomposition(self, feature='Open' , model='additive', period=5, extrapolate_trend='freq'):
        """
        Decomposing time series into its components (trend, seasonality, residual) using moving averages

        Args:
            dataframe (pandas dataframe): time series data
        """
        print('\nDecomposing time series into its components (trend, seasonality, residual) using moving averages\n')
        x_vals = self.data.index
        stock_symbol = self.stock_symbol

        decomposed = seasonal_decompose(self.data[feature], model=model, period=period, extrapolate_trend=extrapolate_trend)
        
        visualizer1 = DataVisual()
        visualizer1.plt_decomposition(x_vals, decomposed, stock_symbol)
        return None
    
    def stationarity(self, feature='Open', significant_val=0.05):
        """
        Assessing if time series is stationary or not by using Advanced Dickey-Fuller test (ADF)
        Null hypothesis: not possible to state whether series is stationary or not
        Alternative hypothesis: series is stationary
        """
        print('\nAssessing if time series is stationary or not by using Advanced Dickey-Fuller test (ADF)\n')
        results = adfuller(self.data[feature], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
        print('Test statistic: {}'.format(results[0]))
        print("MacKinnon's approximate p-value: {}".format(results[1]))
        if results[1] < significant_val:
            print('ADF test shows that time series is stationary.\n')
        else:
            print('ADF test fails to confirm whether time series is stationary or not.\n')
        print('Critical values of the test and confidence levels:\n')
        for key, value in results[4].items():
            print(str(key) + ': ' + str(value))
        return None

    def acf_pacf(self, feature='Open', alpha=0.05):
        """
        Plotting Auto-Correlationf Func and Partial Auto-Correlation Func
        """
        acf_array = acf(self.data[feature], alpha=alpha)
        pacf_array = pacf(self.data[feature], alpha=alpha)

        visualizer = DataVisual()
        visualizer.plt_acf_pacf(self.stock_symbol, acf_array, pacf_array, feature)
        return None
    
    def calculate_daily_return(self, feature='Open'):
        """
        Calculating daily return
        """
        print('\nCalculating daily return\n')
        self.data['daily_return'] = self.data[feature].pct_change()
        return None

