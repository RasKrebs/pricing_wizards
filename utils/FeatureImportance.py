# Data Manipulation
from typing import Type
import pandas as pd

# Scikit learn
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.helpers import plot_feature_importances

class FeatureImportance():

    def __init__(self,
                 results: Type[Bunch] = None):

        # Setting values
        self.results = results

        # Define dictionary for regressor (keys) and feature importances (values)
        feature_importances: dict = dict()

        for _, values in results.items():
            for value in values:
                feature_importances[value.label] = value.feature_importances

        # Transform the feature importances to a pandas dataframe
        self.feature_importances_df = pd.DataFrame({regressor: dict(feature_importances[regressor]) for regressor in feature_importances})

        print("+-------------------------+-----------+")
        print("PERMUTATION IMPORTANCE FROM FEATURES")
        print("+-------------------------+-----------+")
        print(self.feature_importances_df)

    def rank(self) -> pd.DataFrame:
        # Calculate mean for each row (feature)
        self.feature_importances_df['average'] = self.feature_importances_df.mean(axis=1)

        # Add a new column for the rank of the mean values: (ascending=False) due to negative values
        self.feature_importances_df['rank'] = self.feature_importances_df['average'].rank(ascending=False)

        # Generate visualization
        plot_feature_importances(self.feature_importances_df)

        return self.feature_importances_df