"""
Objective: class of WindowGenerator for time series forecasting (source:TensorFlow tutorial)
Author: https://github.com/osevas
Date: 2024-05-15
Reference: https://https://github.com/osevas/2023_2024_spring_derivatives_project
Environment activation: 
C:\\Users\\oseva\\Documents\\py_envs\\deriv_env\\Scripts\\activate
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class WindowGeneratorSerie():
    """creates windowed dataset for time series forecasting
    """
    def __init__(self, train_serie, val_serie, test_serie, window_size, label_size, shift, batch_size, shuffle_buffer):
        # Store the raw data.
        self.train_serie = train_serie
        self.val_serie = val_serie
        self.test_serie = test_serie

        # Work out the window parameters.
        self.window_size = window_size
        self.label_size = label_size
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

        

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def windowed_ds(self, serie):
        """Generates dataset windows

        Args:
          series (array of float) - contains the values of the time series
          window_size (int) - the number of time steps to include in the feature
          batch_size (int) - the batch size
          shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
          dataset (TF Dataset) - TF Dataset containing time windows
        """
      
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(serie)
        
        # Window the data but only take those with the specified size
        dataset = dataset.window(self.window_size + self.label_size, shift=self.shift, drop_remainder=True)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + self.label_size))

        # Create tuples with features and labels 
        dataset = dataset.map(lambda window: (window[:-self.label_size], window[-self.label_size:]))

        # Shuffle the windows
        dataset = dataset.shuffle(self.shuffle_buffer)
        
        # Create batches of windows
        dataset = dataset.batch(self.batch_size).prefetch(1)
        
        return dataset
    
    
    
    def create_datasets(self):
        """
        Function that returns tensorflow dataset

        Returns:
            _type_: pandas dataframe
        """
        # Create window generator
        train_ds = self.windowed_ds(self.train_serie)
        val_ds = self.windowed_ds(self.val_serie)
        test_ds = self.windowed_ds(self.test_serie)
        return train_ds, val_ds, test_ds
    
    def print_prop(self, dataset):
        # Print properties of a single batch
        # Print properties of a single batch
        for windows in dataset.take(1):
            print(f'data type: {type(windows)}')
            print(f'number of elements in the tuple: {len(windows)}')
            print(f'shape of first element: {windows[0].shape}')
            print(f'shape of second element: {windows[1].shape}')
    