"""
Objective: Class for LSTM with attention
Author: https://github.com/osevas
Date: 2024-05-05
"""
import tensorflow as tf
import pandas as pd
import keras
import pickle
import matplotlib.pyplot as plt
from keras import layers, Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from window_generator_serie import WindowGeneratorSerie
from keras.models import load_model
from datetime import timedelta


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
    
    def normalize_data(self, folder_name):
        """
        Function that normalizes data

        Returns:
            _type_: pandas dataframe
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_main[self.feature] = scaler.fit_transform(self.data_main[self.feature].values.reshape(-1, 1))
        # Save the scaler object to a file
        with open('./' + folder_name + '/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
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
    
    def get_tf_dataset(self, folder_name):
        """
        Function that returns tensorflow dataset

        Returns:
            _type_: pandas dataframe
        """
        self.normalize_data(folder_name) # normalize the feature column of the main data
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
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.input_width, 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50, return_sequences=False))  # Only the last time step

        # Adding a Dense layer to match the output shape with y_train
        model.add(Dense(self.days_to_predict))
        print(model.summary())

        return model
    
    def compile_fit(self, model, folder_name):
        """
        Function that compiles and fits the model

        Args:
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_ds, val_ds, test_ds = self.get_tf_dataset(folder_name) # get tensorflow dataset
        
        # Compile
        model.compile(optimizer='adam', loss='mean_squared_error')

        # CALLBACKS
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # Callback to save the model periodically
        model_checkpoint = ModelCheckpoint('./' + folder_name + '/best_model.keras', save_best_only=True, monitor='val_loss')

        # Callback to reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        # Callback for TensorBoard
        tensorboard = TensorBoard(log_dir='./' + folder_name + '/logs')

        # Callback to log details to a CSV file
        csv_logger = CSVLogger('./' + folder_name + '/training_log.csv')

        # Combining all callbacks
        callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

        history = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks_list)

        return history, model
    
    def simulate(self, folder_name):
        """
        Function that simulates the LSTM with attention
        """
        model = self.build1()
        history, model = self.compile_fit(model, folder_name)
        return None
    
    def predict(self, folder_name):
        """
        Function that predicts the LSTM with attention

        1) Load the best model
        2) Predict the test data
        3) transform the data back to original scale
        4) plot the data
        """
        # Load the best model
        best_model = load_model('./' + folder_name + '/best_model.keras')

        # Arrange data to use in prediction
        test_days_predict = self.data_main[self.feature].iloc[(-self.input_width - self.days_to_predict):-self.days_to_predict].values.reshape(-1, 1)
        days_predict = self.data_main[self.feature].iloc[-self.input_width:].values.reshape(-1, 1)

        # Load the scaler object
        with open('./' + folder_name + '/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            test_days_predict = scaler.transform(test_days_predict)
            days_predict = scaler.transform(days_predict)
            y_test_pred = best_model.predict(test_days_predict) # predicting last days_to_predict days of the test data
            y_test_pred = scaler.inverse_transform(y_test_pred)
            y_pred = best_model.predict(days_predict) # predicting the next days_to_predict days
            y_pred = scaler.inverse_transform(y_pred)
        
        plt.figure(figsize=(40,6))
        plt.plot(self.data_main.index[-self.input_width * 3:], self.data_main[self.feature].iloc[-self.input_width * 3:], color='blue')

        # adding new days to index
        new_index = self.data_main.index.append(pd.date_range(self.data_main.index[-1] + timedelta(days=1), periods=self.days_to_predict, freq='D'))
        # new_index = test_index.append(new_index)

        plt.plot(new_index[-self.days_to_predict:], y_pred, marker='o')
        plt.xlabel('Date')
        plt.ylabel(self.feature + 'price' + 'of ' + self.ticker)
        plt.legend(['Training Data', 'LSTM pred'], loc='upper left')
        plt.grid(visible=True, which='both', axis='both')
        plt.savefig('./' + folder_name + '/LSTM_simulation.png')
    
        return None

