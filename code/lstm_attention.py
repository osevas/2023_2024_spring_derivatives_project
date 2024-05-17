"""
Objective: Class for LSTM with attention
Author: https://github.com/osevas
Date: 2024-05-05
"""
import tensorflow as tf
import pandas as pd
import keras
from keras import layers, Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from window_generator_serie import WindowGeneratorSerie

print("TensorFlow version:", tf.__version__)

class LSTMAttention():
    """
    Class for LSTM with attention
    """

    def __init__(self, df_main, ticker, train_day_count, val_day_count, test_day_count, feature, input_width=40, days_to_predict=5):
        """
        Constructor

        Args:
            units (_type_): _description_
            num_classes (_type_): _description_
        """
        self.data_main = df_main
        self.data_main_copy = df_main.copy()
        self.ticker = ticker
        self.train_day_count = train_day_count
        self.val_day_count = val_day_count
        self.test_day_count = test_day_count
        self.feature = feature
        self.input_width = input_width
        self.days_to_predict = days_to_predict

        

    # def call(self, inputs):
    #     """
    #     Function that calls LSTM with attention

    #     Args:
    #         inputs (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     lstm_out, _, _ = self.lstm(inputs)
    #     attention = tf.keras.layers.Attention()([lstm_out, lstm_out])
    #     attention = Flatten()(attention)
    #     output = self.dense(attention)
    #     return output
    
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
        train_df = self.data_main.iloc[-(self.train_day_count + self.test_day_count + self.val_day_count) : -(self.test_day_count + self.val_day_count - 1), :]
        train_index = train_df.index
        train_np = train_df.to_numpy()

        # last_day_price = train_np[-1]

        return train_df, train_index, train_np
    
    def get_val_data(self):
        """
        Function that returns validation data

        Returns:
            _type_: pandas dataframe
        """
        val_df = self.data_main.iloc[-(self.val_day_count + self.test_day_count) : -self.test_day_count, :]
        val_index = val_df.index
        val_np = val_df.to_numpy()
        return val_df, val_index, val_np
    
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
        train_df, _, _ = self.get_train_data()
        val_df, _, _ = self.get_val_data()
        test_df, _, _ = self.get_test_data()

        # Create window generator
        window_gen_1 = WindowGeneratorSerie(train_df[self.feature].values, val_df[self.feature].values, test_df[self.feature].values, self.input_width, 
                                            self.days_to_predict, 1, 32, 1000)
        train_ds, val_ds, test_ds = window_gen_1.create_datasets()
        window_gen_1.print_prop(train_ds)
        return train_ds, val_ds, test_ds

    def build1(self):
        """
        Function that simulates with 3 LSTM layers and 1 Dense layer

        Returns:
            _type_: 
        """
        
        # Model architecture
        model = Sequential()

        # LSTM layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.input_width, self.days_to_predict)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50, return_sequences=False))  # Only the last time step

        # Adding a Dense layer to match the output shape with y_train
        model.add(Dense(self.days_to_predict))
        print(model.summary())

        return model
    
    def compile_fit(self, model):
        """
        Function that compiles and fits the model

        Args:
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_ds, val_ds, test_ds = self.get_tf_dataset() # get tensorflow dataset
        
        # Compile
        model.compile(optimizer='adam', loss='mean_squared_error')

        # CALLBACKS
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # Callback to save the model periodically
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

        # Callback to reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        # Callback for TensorBoard
        tensorboard = TensorBoard(log_dir='./logs')

        # Callback to log details to a CSV file
        csv_logger = CSVLogger('training_log.csv')

        # Combining all callbacks
        callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

        history = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks_list)

        return history, model
    
    def simulate(self):
        """
        Function that simulates the LSTM with attention
        """
        model = self.build1()
        # history, model = self.compile_fit(model)
        return None

