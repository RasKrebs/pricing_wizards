import pandas as pd
from typing import Type
from sklearn.utils import Bunch
from utils.prediction import plot_feature_importances

def rank_feature_importances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank feature importances for each feature across models.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing feature importances for each model.

    Returns:
    - pd.DataFrame: DataFrame with added 'average' and 'rank' columns representing
      the average feature importance and its rank across models, respectively.
    """

    # Create dataframe
    ranked_feature_importances_df = df

    # Calculate mean for each row (feature)
    ranked_feature_importances_df['average'] = ranked_feature_importances_df.mean(axis=1)

    # Add a new column for the rank of the mean values: (ascending=False) due to negative values
    ranked_feature_importances_df['rank'] = ranked_feature_importances_df['average'].rank(ascending=False)

    return ranked_feature_importances_df

def main(prediction_results: Type[Bunch]) -> pd.DataFrame:
    """
    Main function to process prediction results and rank feature importances.

    Parameters:
    - prediction_results (Type[Bunch]): Prediction results containing regressor labels and feature importances.

    Returns:
    - pd.DataFrame: DataFrame with ranked feature importances across different regressors.
    """

    # Define dictionary for regressor (keys) and feature importances (values)
    feature_importances: dict = dict()

    for _, values in prediction_results.items():
        for value in values:
            feature_importances[value.label] = value.feature_importances

    # Transform the feature importances to a pandas dataframe
    feature_importances_df = pd.DataFrame({regressor: dict(feature_importances[regressor]) for regressor in feature_importances})

    # Ranking the feature importances
    ranked_feature_importances = rank_feature_importances(feature_importances_df)

    # Generate visualization
    plot_feature_importances(ranked_feature_importances)

    return ranked_feature_importances