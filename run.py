import pandas as pd
from typing import Type
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from utils.object import MLModelConfig
from scripts.prediction import main as main_prediction
from scripts.preprocessing import main as main_preprocessing
from scripts.feature_importance import main as main_feature_importance

if __name__ == '__main__':
    ## UNCOMMENT THIS LINES IF YOU WANT TO TEST WITH A SAMPLE.CSV
    # df = pd.read_csv('sample.csv')

    # df = df.sample(frac=0.001, replace=True, random_state=1)

    # # Define categorical features
    # categorical_features = ['brand_name', 'category_name']

    # # Extract features and target variable
    # X = df[categorical_features]
    # y = df['listing_price']

    # # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create a configuration object
    # model_config = MLModelConfig(X, y, X_train, X_test, y_train, y_test)

    main_preprocessing() # TODO

    # Running predictions
    prediction_results: Type[Bunch] = main_prediction(model_config)

    main_feature_importance(prediction_results)
