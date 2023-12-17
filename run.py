# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.prediction import Prediction
from utils.FeatureImportance import FeatureImportance
from utils.BaseRegression import BaseRegression

if __name__ == '__main__':
    #### PREPROCESSING - TODO ####

    # Get data
    data = PricingWizardDataset()

    #### LINEAR_REGRESSION ####

    # linear_regression = BaseRegression(data)
    # linear_regression.run()

    #### PREDICTION (SVM + RANDOM FOREST) ####

    # Running SVM and Random Forest
    prediction = Prediction(data)
    prediction_results = prediction.run()

    #### FEATURE IMPORTANCE ####

    feature_importance = FeatureImportance(prediction_results)

    # Ranking feature importances
    feature_importance.rank()
