import pandas as pd

class DataLoader:
    """
    Class for loading data.
    If train and test paths are provided, they are loaded separately.
    If only one file is provided, it's treated as the main dataset.
    """
    def __init__(self, train_path=None, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """
        Load data from specified paths.
        
        Returns:
            tuple: (train_data, test_data) - pandas DataFrames
        """
        if self.train_path:
            self.train_data = pd.read_csv(self.train_path)
            print(f"Loaded train dataset: {self.train_path}")
        if self.test_path:
            self.test_data = pd.read_csv(self.test_path)
            print(f"Loaded test dataset: {self.test_path}")
        if self.train_data is None and self.test_data is None:
            raise ValueError("No data paths provided for loading.")
        return self.train_data, self.test_data 