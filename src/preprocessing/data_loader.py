import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Class for loading data.
    If train and test paths are provided, they are loaded separately.
    If only one file is provided, it's automatically split into train and test sets.
    """
    def __init__(self, train_path=None, test_path=None, test_size=0.2, random_state=42):
        """
        Initialize DataLoader.
        
        Args:
            train_path: str, path to training data or full dataset
            test_path: str or None, path to test data
            test_size: float, proportion of data to use for testing when splitting
            random_state: int, random state for reproducibility
        """
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """
        Load data from specified paths.
        If only train_path is provided, automatically split into train and test sets.
        
        Returns:
            tuple: (train_data, test_data) - pandas DataFrames
        """
        if self.train_path and self.test_path:
            # Load separate train and test datasets
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            print(f"Loaded train dataset: {self.train_path}")
            print(f"Loaded test dataset: {self.test_path}")
        elif self.train_path:
            # Load single dataset and split
            full_data = pd.read_csv(self.train_path)
            print(f"Loaded full dataset: {self.train_path}")
            print(f"Splitting into train ({1-self.test_size:.0%}) and test ({self.test_size:.0%}) sets...")
            
            self.train_data, self.test_data = train_test_split(
                full_data,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )
            print(f"Split complete. Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        else:
            raise ValueError("No data paths provided for loading.")
            
        return self.train_data, self.test_data 