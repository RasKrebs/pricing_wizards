# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.Prediction import Prediction
from utils.FeatureImportance import FeatureImportance
from utils.BaseRegression import BaseRegression

if __name__ == '__main__':
    #### PREPROCESSING - TODO ####

    # Get data
    data = PricingWizardDataset()

    #### LINEAR_REGRESSION ####

    linear_regression = BaseRegression(data)
    linear_regression.run()

    #### PREDICTION (SVM + RANDOM FOREST) ####

    # Working with a sample
    ## COMMENT the following line if you don't want to work with a sample
    data.df = data.df.sample(frac=0.003, replace=True, random_state=1)

    # Define features to work with
    features = ['listing_price', 'brand_name', 'category_name', 'condition_name', 'viewed_count', 'subcategory_name']

    # Use only selected features
    data.df = data.df[features]

    # Running SVM and Random Forest
    prediction = Prediction(data)
    prediction_results = prediction.run()

    # Feature Importance
    feature_importance = FeatureImportance(prediction_results)

    # Ranking feature importances
    feature_importance.rank()
