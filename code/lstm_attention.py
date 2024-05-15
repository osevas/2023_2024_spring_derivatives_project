"""
Objective: Class for LSTM with attention
Author: https://github.com/osevas
Date: 2024-05-05
"""
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from window_generator import WindowGenerator
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

class LSTMAttention(Model):
    """
    Class for LSTM with attention
    """

    def __init__(self, df_main, ticker, train_day_count, test_day_count, feature, input_width=40, days_to_predict=5):
        """
        Constructor

        Args:
            units (_type_): _description_
            num_classes (_type_): _description_
        """
        super(LSTMAttention, self).__init__()
        self.data_main = df_main
        self.data_main_copy = df_main.copy()
        self.ticker = ticker
        self.train_day_count = train_day_count
        self.test_day_count = test_day_count
        self.feature = feature
        self.input_width = input_width
        self.days_to_predict = days_to_predict

        print("LSTMAttention class initialized")
        
        # self.lstm = LSTM(self.units, return_sequences=True, return_state=True)
        # self.dense = Dense(num_classes)

    def call(self, inputs):
        """
        Function that calls LSTM with attention

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        lstm_out, _, _ = self.lstm(inputs)
        attention = tf.keras.layers.Attention()([lstm_out, lstm_out])
        attention = Flatten()(attention)
        output = self.dense(attention)
        return output
    
    def normalize_data(self):
        """
        Function that normalizes data

        Returns:
            _type_: pandas dataframe
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_main[self.feature] = scaler.fit_transform(self.data_main[self.feature].values.reshape(-1, 1))
        return None

    def get_train_data(self):
        """
        Function that returns training data

        Returns:
            _type_: pandas dataframe
        """
        train_df = self.data_main.iloc[-(self.train_day_count + self.test_day_count) : -(self.test_day_count - 1), :]
        train_index = train_df.index
        train_np = train_df.to_numpy()

        # last_day_price = train_np[-1]

        return train_df, train_index, train_np
    
    def get_test_data(self):
        """
        Function that returns testing data

        Returns:
            _type_: pandas dataframe
        """
        test_df = self.data_main.iloc[-self.test_day_count:, :]
        test_index = test_df.index
        test_np = test_df.to_numpy()
        return test_df, test_index, test_np
    
    def get_tf_dataset(self):
        """
        Function that returns tensorflow dataset

        Returns:
            _type_: pandas dataframe
        """
        self.normalize_data() # normalize the feature column of the main data
        train_df, train_index, _ = self.get_train_data()
        test_df, test_index, _ = self.get_test_data()

        # Create window generator
        window_gen_1 = WindowGenerator(input_width=self.input_width, label_width=self.days_to_predict, shift=self.days_to_predict, train_df=train_df, val_df=test_df, test_df=test_df, label_columns=[self.feature])
        print(window_gen_1)
        return None

    def simulate(self):
        """
        Function that simulates

        Returns:
            _type_: pandas dataframe
        """
        self.get_tf_dataset()
        return None

