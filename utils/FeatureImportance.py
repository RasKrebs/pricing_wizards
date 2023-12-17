# Data Manipulation
import pandas as pd

def set_feature_importances(results):

    # Define dictionary for regressor (keys) and feature importances (values)
    feature_importances: dict = dict()

    for _, values in results.items():
        for value in values:
            feature_importances[value.label] = value.feature_importances

    # Transform the feature importances to a pandas dataframe
    feature_importances_df = pd.DataFrame({regressor: dict(feature_importances[regressor]) for regressor in feature_importances})

    print("+-------------------------+-----------+")
    print("PERMUTATION IMPORTANCE FROM FEATURES")
    print("+-------------------------+-----------+")
    print(feature_importances_df)

def rank_feature_immportances(df) -> pd.DataFrame:
    # Calculate mean for each row (feature)
    df['average'] = df.mean(axis=1)

    # Add a new column for the rank of the mean values: (ascending=False) due to negative values
    df['rank'] = df['average'].rank(ascending=False)

    return df