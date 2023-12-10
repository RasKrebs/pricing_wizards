import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import warnings

class PricingWizardDataset:
    def __init__(self,
                 *,
                 filename:str = "post_preprocessing_without_dummies.csv",
                 alternative_data_path:str = None,
                 train_size:int = .8,
                 test_size:int = .2,
                 ramdom_state = 42,
                 outlier_removal = True) -> None:
        """
        Data extraction and partioning class.

        Class for extracting Trendsales data for modeling in Pricing Wizards exam Project. The class should simplify the process of passing data to models.

        Args:
            filename (str, optional): Filename from subfolder to load. Defaults to post_preprocessing_without_dummies
            alternative_path (str, optional): In case an alternative path is prefered, such can be called. Defaults to None.
            train_size (float, optional): Portion of dataset to include in train split. Defaults to .8 or 80% of data.
            test_size (float, optional): Portion of dataset to include in test split. This subset will per default be a holdout dataset, and should only be used for final model performance. 
                                         Validation subsetes will be extracted from the training set. This ensures sufficient large test set. Defaults to .2 or 20% of data.
            random_state = (int, optional): Controls the behaviour of shuffling applied to the data before applying the split. Enables reproducibaility of resutls across multiple initalizations. Defaults to 42.
            
        
        Attributes:
            data_directory (str): Full path to subdirectory storing data 
            filename (str): Name of data csv file to be extracted
            seed (int): Seed for reproducability

        Methods:
            __call__: Calling object after initalization will return partitioned dataset, in X_train, X_test, Y_train, Y_test format.
        """
        
        # Data storage details
        if alternative_data_path:
            self.data_folder = alternative_data_path
        else:
            # Control the working directories are correct
            assert 'pricing_wizards' in os.getcwd(), f"This program can only be executed inside the pricing_wizards directory if no alternative data path is specified. You're currently in {os.getcwd()}. Please change directory into pricing wizards before you calling class or specify an alternative data path."
            
            self.data_folder = os.path.join(os.getcwd().split('pricing_wizards')[0],'pricing_wizards/data/')
        
        # Name of data file
        self.filename = filename + '.csv' if '.csv' not in filename else filename
        
        # Asserting file exists in data folder
        assert self.filename in os.listdir(self.data_folder), f'File, {self.filename}.csv, does not appear in data folder, {self.data_folder}. Please make sure the correct filename and data folder is specified. {os.listdir(self.data_folder)}'
        
        # Seed for reproducability
        self.seed = ramdom_state
        np.random.seed(self.seed)
        
    
        # Splitting details
        self.train_size = train_size
        self.test_size = test_size
        
        # If None is not passed as argument
        if not self.train_size:
            assert self.train_size + self.test_size + self.val_size == 1, "Sum of split sizes must equal 1. Ensure passed size arguments is equal to 1"
            

        # Loading data
        self.__load__()
        
        # Outlier Removal, if specified
        if outlier_removal: 
            self.delete_outliers()
        
        self.outlier_removed = outlier_removal

        # UX Feedback
        self.__repr = f'Dataset Loaded: {self.filename.rstrip(".csv")}\n\tNumber of Rows: {len(self.df)}\n\tOutlier Removal: {self.outlier_removed}\n\tTrain Size: {self.train_size}\n\tTest Size: {self.test_size}\n\tRandom State: {self.seed}'
        print(self.__repr)
    
    def __load__(self):
        """Internal class method for loading data from data folder"""
        
        # Loading data using pandas
        self.df = pd.read_csv(f'{self.data_folder}{self.filename}')
        
        # Fill null brand values with Unassigned
        self.df.brand_name = self.df.brand_name.fillna('Unassigned')
    
    def delete_outliers(self):
        """
        Method for removing outliers. 
        
        Method used is a IQR performed on a logscaled version of the listing price. Using a logscaled listing price version helps seperate brand listings prices in regards to both upper and lower bounds.
        This method will overwrite the df class attribute with the a dataset that does not have any outliers. The original dataset will be assigned `self.raw_df`.
        """
        
        # Generating dataset copy
        data = self.df.copy()
        
        # Logscaled Listing Price
        data['log_price'] = np.log(data['listing_price']) 
        
        # Assigning Unassigned to nulls
        data.brand_name = data.brand_name.fillna('Unassigned')
        
        # Computing qunatiles on brandlevel
        iqr_brand = data.groupby('brand_name').agg(q25 = ('log_price', lambda x: x.quantile(.25)),
                                                      q75 = ('log_price', lambda x: x.quantile(.75))).reset_index()

        # Computing difference and thresholds
        iqr_brand = iqr_brand.assign(
            difference = lambda x: x.q75 - x.q25,
            upper_bound = lambda x: x.q75 + 2*(x.difference), # Usually it is 1.5*(q3-q1), but given the nature of this dataset, the thresholds have been increased to make it less discriminative
            lower_bound = lambda x: x.q25 - 2*(x.difference))

        # Merge listings with brand quantile bounds
        data  = data[['classified_id','log_price','brand_name']].merge(iqr_brand, on='brand_name',how='left')     
        
        # Determine if listing is outlier
        self.df['outlier'] = (data.log_price > data.upper_bound) | (data.log_price < data.lower_bound)
        
        # Created raw df attribute before deleting rows (including outlier column)
        self.raw_df = self.df.copy()
        
        # Overwriting df attribute
        self.df = self.df[self.df.outlier == False]
    
    
    def stratify_train_test_split(self, 
                                  val_size=.2,
                                  split_based_on_original_size = True):
        """
        Method for splitting data into train, test and validation subsets.
        
        Args:
            val_size (float, optional): Specifies the proportion of the dataset that should be saved for validation. Defaults to .2.
            split_based_on_original_size (bool, optional): Specifies wether val_size is based on original dataset size or for training size. If True, the validation split will equal 20% of the original dataset size. Defaults to True.
        
        Return:
        (X_train, X_test, X_val), (y_train, y_test, y_val)
        """
        
        # Lambda function for computing bins for listing prices
        bins = lambda x: pd.cut(x, 4)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(columns='listing_price'), 
                                                            self.df.listing_price.to_numpy(), 
                                                            stratify=bins(self.df.listing_price),
                                                            train_size=self.train_size,
                                                            test_size=self.test_size, 
                                                            random_state=self.seed)
        # Train val split
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          stratify=bins(y_train),
                                                          test_size=val_size/self.train_size if split_based_on_original_size else val_size,
                                                          random_state=self.seed)
        
        # Perform KS Test
        equally_distributed_subsets = self.__ks_test__(y_train, y_test, y_val, alpha=.05)
        
        if not equally_distributed_subsets: 
            warnings.warn('Warning..... KS test for subsets failed. Distribution of splits are different')
        else:
            print('Dependent variable distribution is equal across all subsets')
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    
    def __repr__(self) -> str:
        return self.__repr
    
    def __ks_test__(self, train, test, val, alpha=.05) -> bool:
        """Internal class method for Kolmogorov-Smirnov test. Used to evaluate if distributions in train, test and val are similar. Shoud return true if stratifeid split worked as intendede"""
        
        # Perform Kolmogorov-Smirnov for all splits
        _, train_val = stats.ks_2samp(train, val)
        _, test_val = stats.ks_2samp(test, val)
        _, train_test = stats.ks_2samp(test, train)

        # Return boolean value of wether or not all values are 3 or not
        return sum([alpha < p_val for p_val in [train_val, test_val, train_test]]) == 3
        