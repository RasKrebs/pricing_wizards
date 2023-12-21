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
                 preprocess = True,
                 outlier_iqr_scale = 2) -> None:
        """
        Data loading, preprocessing and subsetting class.

        Class for extracting Trendsales data for modeling in Pricing Wizards exam Project. The class should simplify the process of passing data to models.

        Args:
            filename (str, optional): Filename from subfolder to load. Defaults to post_preprocessing_without_dummies
            alternative_path (str, optional): In case an alternative path is prefered, such can be called. Defaults to None.
            train_size (float, optional): Portion of dataset to include in train split. Defaults to .8 or 80% of data.
            test_size (float, optional): Portion of dataset to include in test split. This subset will per default be a holdout dataset, and should only be used for final model performance.
                                         Validation subsetes will be extracted from the training set. This ensures sufficient large test set. Defaults to .2 or 20% of data.
            random_state (int, optional): Controls the behaviour of shuffling applied to the data before applying the split. Enables reproducibaility of resutls across multiple initalizations. Defaults to 42.
            preprocess (bool, optional): Determines wether or not to preprocess data. If True, the dataset will be preprocessed. Defaults to True.


        Relevant Attributes:
            df (pd.DataFrame): Retuns the dataset in a pandas dataframe format
            columns (list): Returns the columns of the dataset
            index (list): Returns the index of the dataset
            shape (list): Returns the shape of the dataset
            dtypes (list): Returns the datatypes of the dataset
            outlier_removed (bool): Returns True if outliers have been removed, else False
            raw_df (pd.DataFrame): Returns the raw dataset before preprocessing
            __repr__ (str): Returns a string with information about the dataset. This is printed when the class is called.
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

        # Outlier Removal and general preprocessing, if specified
        self.preprocess = preprocess
        self.outlier_scaler = outlier_iqr_scale
        if preprocess:
            self.process()
            self.outlier_removed = True
        else:
            self.outlier_removed = False

        # Pandas Dataframe Variables for easier use
        self.columns = self.__columns__()
        self.index = self.__index__()
        self.shape = self.__shape__()
        self.dtypes = self.__dtypes__()

        # Functions applied
        self.functions_applied = []

        # UX Feedback
        self.__repr = f'Dataset Loaded: {self.filename.replace(".csv", "")}\n\tNumber of Rows: {self.df.shape[0]}\n\tNumber of Columns: {self.df.shape[1]}\n\tOutlier Removal: {self.outlier_removed}\n\tTrain Size: {self.train_size}\n\tTest Size: {self.test_size}\n\tRandom State: {self.seed}'
        print(self.__repr)

    def __load__(self) -> None:
        """Internal class method for loading data from data folder"""

        # Loading data using pandas
        self.df = pd.read_csv(f'{self.data_folder}{self.filename}')

        # Fill null brand values with Unassigned
        self.df.brand_name = self.df.brand_name.fillna('Unassigned')

    def __columns__(self) -> list:
        """Internal class method for returning columns"""
        return self.df.columns

    def __index__(self) -> list:
        """Internal class method for returning index"""
        return self.df.index

    def __shape__(self) -> list:
            """Internal class method for returning shape of dateframe"""
            return self.df.shape

    def __dtypes__(self) -> list:
            """Internal class method for returning shape of dateframe"""
            return self.df.dtypes

    def process(self) -> None:
        """
        Method for processing data.

        Method used is a IQR performed on a logscaled version of the listing price. Using a logscaled listing price version helps seperate brand listings prices in regards to both upper and lower bounds.
        This method will overwrite the df class attribute with the a dataset that does not have any outliers. The original dataset will be assigned `self.raw_df`.
        """

        # Filling Brand Names with Unassigned + subsubsubcategory_name
        self.df.brand_name = self.df.brand_name.fillna('Unassigned')
        self.df.loc[self.df[self.df.brand_name == 'Unassigned'].index, 'brand_name'] = self.df[self.df.brand_name == 'Unassigned'].brand_name + '_' + self.df[self.df.brand_name == 'Unassigned'].subsubsubcategory_name

        # Log Transforming Numeric Variables (+ 1 to avoid log(0)
        self.df['log_listing_price'] = np.log(self.df['listing_price'] + 1)
        self.df['log_viewed_count'] = np.log(self.df['viewed_count'] + 1)

        # Saving raw copy
        self.raw_df = self.df.copy()

        self.outlier_removal()


    def outlier_removal(self) -> None:
        """Method for removing outliers. This method will overwrite the df class attribute with the a dataset that does not have any outliers. The original dataset will be assigned `self.raw_df`."""

        # Copying data
        data = self.df.copy()


        # Computing qunatiles on brandlevel
        iqr_brand = data.groupby('brand_name').agg(q25 = ('log_listing_price', lambda x: x.quantile(.25)),
                                                      q75 = ('log_listing_price', lambda x: x.quantile(.75))).reset_index()

        # Computing difference and thresholds
        iqr_brand = iqr_brand.assign(
            difference = lambda x: x.q75 - x.q25,
            upper_bound = lambda x: x.q75 + self.outlier_scaler*(x.difference), # Usually it is 1.5*(q3-q1), but given the nature of this dataset, the thresholds have been increased to make it less discriminative
            lower_bound = lambda x: x.q25 - self.outlier_scaler*(x.difference))


        # Merge listings with brand quantile bounds
        data  = data[['classified_id','log_listing_price','brand_name']].merge(iqr_brand, on='brand_name',how='left')

        # Determine if listing is outlier
        data['outlier'] = (data.log_listing_price > data.upper_bound) | (data.log_listing_price < data.lower_bound)

        # Overwriting df attribute
        self.df = self.df[self.df.classified_id.isin(data[data.outlier == False].classified_id)]

        # Removing viewed count above 99.5 percentile
        self.df = self.df[self.df.viewed_count <= self.df.viewed_count.quantile(.995)]


    def apply_function(self, function, *args, **kwargs) -> None:
        """Method for applying user defined function to dataset"""
        self.df = function(self.df, *args, **kwargs)

        # Append function name to list of functions applied
        self.functions_applied.append(function.__globals__['__name__'])


    def stratify_train_test_split(self,
                                  independent_variables: list = None,
                                  y_column: str = 'listing_price',
                                  val_size=.2,
                                  dtypes_to_exclude: list = None,
                                  split_based_on_original_size = True,
                                  return_splits = True) -> tuple:
        """
        Method for splitting data into train, test and validation subsets.

        Args:
            independent_variables (list, optional): Specifies which columns include in split. If none specified, uses all. Defaults to None.
            y_column (str, optional): Specifies which column to use as dependent variable in split. Defaults to listing_price.
            val_size (float, optional): Specifies the proportion of the dataset that should be saved for validation. Defaults to .2.
            split_based_on_original_size (bool, optional): Specifies wether val_size is based on original dataset size or for training size. If True, the validation split will equal 20% of the original dataset size. Defaults to True.
            return_splits (bool, optional): Determines wether or not to return splits or just save them as variables. Defaults to True.

        Return:
        (X_train, X_test, X_val), (y_train, y_test, y_val)
        """

        # Lambda function for computing bins for listing prices
        bins = lambda x: pd.cut(x, 4)

        # Data for stratified split
        data = self.df.copy()

        # If no columns are specified, use all columns
        if not independent_variables:
            independent_variables = self.df.columns

        # If columns are specified, ensure they are in list format
        independent_variables = list(independent_variables)

        # In case dependent variable is passed along with independent variables, remove dependent variable from independent variables
        if y_column in independent_variables:
            independent_variables.remove(y_column)

        # If no dtypes are specified, use all dtypes
        if dtypes_to_exclude:
            data = data.select_dtypes(exclude=dtypes_to_exclude)

        # Assigning X and y
        self.X = data[independent_variables]
        self.y = data[y_column].to_numpy()

        # Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                stratify=bins(data[y_column].to_numpy()),
                                                                                train_size=self.train_size,
                                                                                test_size=self.test_size,
                                                                                random_state=self.seed)
        # If no validation size is specified, return train and test splits
        if val_size == 0:
            if not stats.ks_2samp(self.y_test, self.y_train):
                warnings.warn('Warning..... KS test for subsets failed. Distribution of splits are different')
            else:
                print('Dependent variable distribution is equal across all subsets')

            # Return splits if specified
            if return_splits:
                return self.X_train, self.X_test, self.y_train, self.y_test
            else:
                return

        else:
            # Train val split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                                  self.y_train,
                                                                                  stratify=bins(self.y_train),
                                                                                  test_size=val_size/self.train_size if split_based_on_original_size else val_size,
                                                                                  random_state=self.seed)

            # Perform KS Test
            equally_distributed_subsets = self.__ks_test__(self.y_train, self.y_test, self.y_val, alpha=.05)

            if not equally_distributed_subsets:
                warnings.warn('Warning..... KS test for subsets failed. Distribution of splits are different')
            else:
                print('Dependent variable distribution is equal across all subsets')

            # Return splits if specified
            if return_splits:
                return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val
            else:
                return


    def head(self, n=5) -> pd.DataFrame:
        """Method for returning head of dataset"""
        return self.df.head(n)


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


    def reset_dataset(self) -> None:
        """Method for resetting dataset to original state"""
        # Resetting df to original state and removing functions applied
        self.df = self.raw_df.copy()
        self.functions_applied = []

        # Delete splits
        try:
            del self.X_train
            del self.X_test
            del self.y_train
            del self.y_test
            del self.X_val
            del self.y_val
        except:
            pass

        # Process data if process is True
        if self.preprocess:
            self.process()