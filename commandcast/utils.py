# Tidal Pulse Utility Functions
from dateutil.relativedelta import relativedelta
import pandas as pd

# Used as seasonality input for Nixtla StatsForecast models
# TODO: Use Nixtla's multi-seasonality models and find a way to incorporate
# multiple seasonalities based on frequency (or based on additional user input)
def get_season_length(freq):
    s = 7
    if freq in ['H', 'BH']:
        s = 24
    elif freq in ['T', 'min']:
        s = 60
    elif freq == 'S':
        s = 60
        
    # TODO: Find best season length for yearly data
    elif freq in ['A', 'Y', 'BA', 'BY', 'AS', 'YS', 'BAS', 'BYS']:
        s = 1
    elif freq in ['W', 'SM', 'SMS']:
        s = 52
    elif freq in ['M', 'BM', 'CBM', 'MS', 'BMS', 'CBMS', 'Q', 'BQ', 'QS', 'BQS']:
        s = 12
    return s

def get_freq_str(freq):
    converted_freq = 'days'
    if freq in ['H', 'BH']:
        converted_freq = 'hours'
    elif freq in ['T', 'min']:
        converted_freq = 'minutes'
    elif freq == 'S':
        converted_freq = 'seconds'
    elif freq in ['A', 'Y', 'BA', 'BY', 'AS', 'YS', 'BAS', 'BYS']:
        converted_freq = 'years'
    elif freq in ['W', 'SM', 'SMS']:
        converted_freq = 'weeks'
    elif freq in ['M', 'BM', 'CBM', 'MS', 'BMS', 'CBMS', 'Q', 'BQ', 'QS', 'BQS']:
        converted_freq = 'months'
    return converted_freq

# Adapted from this blog post: https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8
class TimeseriesCV:
    """
    Temporal Cross-Validation Class for time-series data.

    Parameters
    ----------
    train_period : int
        Number of time units to include in each train set. Default is 30.
    test_period : int
        Number of time units to include in each test set. Default is 7.
    freq : str
        Frequency of input parameters, e.g., 'days', 'months', 'years', 'weeks', 'hours', 'minutes', 'seconds'.
        Default is 'days'.

    Methods
    -------
    split(data, validation_split_date=None, date_column='record_date', gap=0)
        Generates train/test split indices for the given DataFrame.
    get_n_splits()
        Returns the number of splits.
    """

    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq
        self.n_splits = 0

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        data : pandas DataFrame
            Data containing at least one column for the record date.
        validation_split_date : datetime-like, optional
            The first date to perform the splitting on.
            If not provided, it will default to the minimum date in the data after the first training set.
        date_column : str, default='record_date'
            Name of the date column in the DataFrame.
        gap : int, default=0
            Number of time units to leave between the train and test sets.

        Returns
        -------
        list of tuples
            Each tuple contains train and test indices (train_indices, test_indices).
        """

        # Validate parameters
        if date_column not in data.columns:
            raise KeyError(f"'{date_column}' not found in data columns.")
        
        data[date_column] = pd.to_datetime(data[date_column])

        # Initialize train/test indices lists
        train_indices_list = []
        test_indices_list = []

        # Set initial validation split date
        if validation_split_date is None:
            validation_split_date = data[date_column].min() + relativedelta(**{self.freq: self.train_period})

        start_train = validation_split_date - relativedelta(**{self.freq: self.train_period})
        end_train = start_train + relativedelta(**{self.freq: self.train_period})
        start_test = end_train + relativedelta(**{self.freq: gap})
        end_test = start_test + relativedelta(**{self.freq: self.test_period})

        # Perform splits
        while end_test < data[date_column].max():
            # Get current train and test indices
            train_indices = data[(data[date_column] >= start_train) & (data[date_column] < end_train)].index.tolist()
            test_indices = data[(data[date_column] >= start_test) & (data[date_column] < end_test)].index.tolist()

            # Append to lists
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)

            # Update start/end dates for the next split
            start_train += relativedelta(**{self.freq: self.test_period})
            end_train = start_train + relativedelta(**{self.freq: self.train_period})
            start_test = end_train + relativedelta(**{self.freq: gap})
            end_test = start_test + relativedelta(**{self.freq: self.test_period})

        # Handle any remaining data in the final split
        if start_test < data[date_column].max():
            train_indices = data[(data[date_column] >= start_train) & (data[date_column] < end_train)].index.tolist()
            test_indices = data[(data[date_column] >= start_test)].index.tolist()
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)

        # Store the number of splits and return indices
        self.n_splits = len(train_indices_list)
        return list(zip(train_indices_list, test_indices_list))

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

# Example Usage
# date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
# df = pd.DataFrame({'record_date': date_range, 'value': range(len(date_range))})

# # Initialize TemporalCV with default settings
# temporal_cv = TimeseriesCV(train_period=30, test_period=7, freq='days')

# # Get train/test splits
# splits = temporal_cv.split(df, date_column='record_date')

# # Display the splits
# for idx, (train_idx, test_idx) in enumerate(splits):
#     print(f"Split {idx + 1}:")
#     print(f"  Train indices: {train_idx}")
#     print(f"  Test indices: {test_idx}")