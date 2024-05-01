"""
Objective: to create an object for visualization
Author: https://github.com/osevas
Date: 2023-03-26
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class DataVisual:
    """
    Objective: to create an object for visualization
    """
    def plt_time_series(self, df, stock_name, feature='High'):
        """
        Plotting stock parameter vs. time/date
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.index, y=df[feature])
        )
        fig.update_layout(yaxis_title = stock_name, xaxis_title='Date')
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        return None
    
    def plt_candlestick(self, df, stock_name):
        """
        Plotting candlestick graph for stock vs time/date

        Args:
            df (pandas dataframe): stock data
            stock_name (string): stock ticker and name
        """
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )
        )
        fig.update_layout(yaxis_title = stock_name, xaxis_title='Date')
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        return None
    
    def plt_ohlc(self, df, stock_name):
        """
        Plotting OHLC graph for stock vs time/date

        Args:
            df (pandas dataframe): stock data
            stock_name (string): stock ticker and name
        """
        fig = go.Figure()
        fig.add_trace(
            go.Ohlc(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )
        )
        fig.update_layout(yaxis_title = stock_name, xaxis_title='Date')
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        return None
    
    def plt_combine(self, df, stock_name):
        """
        Plotting plt_time_series, plt_candlestick and plt_ohlc together
        """
        self.plt_time_series(df, stock_name)
        self.plt_candlestick(df, stock_name)
        self.plt_ohlc(df, stock_name)
        return None
    
    def plt_decomposition(self, x_vals, decomposed, stock_symbol):
        """
        Plotting DecomposeResult of statsmodels.tsa.seasonal.seasonal_decompose.  This is used by DataAnalyzer.

        Args:
            decomposed (DecomposeResult): statsmodels.tsa.seasonal.seasonal_decompose
        """
        fig = make_subplots(rows=5, cols=1, 
            subplot_titles=('Observed', 'Seasonal', 'Trend', 'Residual', 'Weights'))
        y_list = [decomposed.observed, decomposed.seasonal, decomposed.trend, decomposed.resid, decomposed.weights]

        for i, y_item in enumerate(y_list):
            fig.add_trace(go.Scatter(x=x_vals, y=y_item),
                row=i+1, col=1
            )

        fig.update_layout(title_text='Seasonal Decomposition of Stock ' + stock_symbol)
        fig.show()
        return None
    
    def plt_acf_pacf(self, stock_name, acf_array, pacf_array, feature):
        """
        Plotting Auto-Correlation Function and Partial ACF

        Args:
            dataframe (pandas dataframe): stock data
            stock_name (str): stock name
        """
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=('Auto-Correlation Function for ' + stock_name + ' feature ' + feature, 'Partial Auto-Correlation Function for ' + stock_name + ' feature ' + feature))
        
        corr_arrays = [acf_array, pacf_array]

        for i, corr_array in enumerate(corr_arrays):
            lower_y = corr_array[1][:,0] - corr_array[0]
            upper_y = corr_array[1][:,1] - corr_array[0]


            for x in range(len(corr_array[0])):
                fig.add_trace(go.Scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines', line_color='#3f3f3f'), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4', marker_size=12), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)'), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)', fill='tonexty', line_color='rgba(255,255,255,0)'), row=1, col=i+1)
            fig.update_traces(showlegend=False)
            fig.update_xaxes(range=[-1,42])
            fig.update_yaxes(zerolinecolor='#000000')
            
            
        fig.show()
        return None





