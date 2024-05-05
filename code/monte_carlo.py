"""
Objective: Class for Monte Carlo simulation
Author: https://github.com/osevas
Date: 2024-05-05
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import timedelta

class MonteCarlo:
    """
    Class for Monte Carlo simulation
    """

    def __init__(self, data, ticker, train_day_count, test_day_count, feature, num_simulations, days_to_predict=5):
        """
        Constructor

        Args:
            data (_type_): _description_
            ticker (_type_): _description_
        """
        self.data = data
        self.ticker = ticker
        self.train_day_count = train_day_count
        self.test_day_count = test_day_count
        self.feature = feature
        self.num_simulations = num_simulations
        self.days_to_predict = days_to_predict

    def get_train_data(self):
        """
        Function that returns training data

        Returns:
            _type_: pandas dataframe
        """
        train_df = self.data[self.feature].iloc[-(self.train_day_count + self.test_day_count) : -(self.test_day_count - 1)]
        train_index = train_df.index
        train_np = train_df.to_numpy()

        last_day_price = train_np[-1]

        return train_df, train_index, train_np, last_day_price
    
    def get_test_data(self):
        """
        Function that returns testing data

        Returns:
            _type_: pandas dataframe
        """
        test_df = self.data[self.feature].iloc[-self.test_day_count:]
        test_index = test_df.index
        test_np = test_df.to_numpy()
        return test_df, test_index, test_np
    
    def calculate_drift(self):
        """
        Function that calculates parameters

        Returns:
            _type_:
        """
        _, _, train_np, last_day_price = self.get_train_data()
        log_return = np.diff(np.log(train_np))
        avg_daily_return = np.mean(log_return)
        variance = np.var(log_return)
        std_dev = np.std(log_return)
        drift = avg_daily_return - (0.5 * variance)
        return drift, std_dev, last_day_price
    
    def calculate_next_day_price(self):
        """
        Function that calculates next day price

        Args:
            last_day_price (_type_): _description_

        Returns:
            _type_:
        """
        drift, std_dev, last_day_price = self.calculate_drift()
        random_num = norm.ppf(np.random.rand(self.test_day_count + self.days_to_predict, self.num_simulations))
        daily_returns = np.exp(drift + std_dev * random_num)
        next_day_price = np.zeros_like(daily_returns)
        for i in range(0, self.test_day_count):
            next_day_price[i] = last_day_price * daily_returns[i]
            last_day_price = next_day_price[i]
        return next_day_price
    
    def plot_simulation(self, next_day_price):
        """
        Function that plots the simulation
        """
        _, train_index, train_np, _ = self.get_train_data()
        _, test_index, _ = self.get_test_data()

        plt.figure(figsize=(40,6))
        plt.plot(train_index[-50:], train_np[-50:])
        plt.plot(test_index, next_day_price[:self.test_day_count])
        plt.savefig('MC_all_simulations.png')
        return None
    
    def best_simulation(self, test_arr, sims_arr):
        """
        Function that returns the best simulation

        Args:
            test_arr (_type_): _description_
            sims_arr (_type_): _description_

        Returns:
            _type_:
        """
        test_arr = np.array(test_arr)
        sims_arr = np.array(sims_arr)
        
        best_sim = np.argmin(np.sum(np.abs(np.subtract(np.repeat(test_arr.reshape((-1, 1)), sims_arr.shape[1], axis=1), sims_arr)), axis=0))
        return best_sim
    
    def plot_best_simulation(self, next_day_price):
        """
        Function that plots the best simulation
        """
        train_df, train_index, train_np, last_day_price = self.get_train_data()
        test_df, test_index, test_np = self.get_test_data()

        best_sim = self.best_simulation(test_np, next_day_price[:self.test_day_count])

        plt.figure(figsize=(40,6))
        plt.plot(train_index[-50:], train_np[-50:])
        plt.plot(test_index, test_np, color='red')

        new_index = test_index + timedelta(days=self.days_to_predict)
        new_index = test_index.append(new_index)
        plt.plot(new_index, next_day_price[:, best_sim])
        
        plt.savefig('MC_best_simulation.png')
        return None
    
    def simulate(self):
        """
        Function that simulates the data

        Returns:
            _type_: pandas dataframe
        """
        # train_df, train_index, train_np = self.get_train_data()
        # test_df, test_index, test_np = self.get_test_data()
        # print(train_df.head())
        # print(train_np)
        next_day_price = self.calculate_next_day_price()
        # print(next_day_price)
        self.plot_simulation(next_day_price)
        self.plot_best_simulation(next_day_price)
        return next_day_price
