# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from scripts.svm import run_svm
from scripts.random_forest import run_rf
from utils.Dataloader import PricingWizardDataset
from utils.helpers import plot_actual_predicted, plot_residuals, plot_model_evaluation, plot_training_time
from utils.DataTransformation import base_regression_pipeline

class Prediction:
    def __init__(self,
                data: PricingWizardDataset = None):

        # Setting values
        self.data = data

        # Encoding the data
        self._transform_data()

        # Get train and test data
        (self.X_train,
         self.X_test,
         self.X_val,
         self.y_train,
         self.y_test,
         self.y_val) = self.data.stratify_train_test_split(
             y_column='log_listing_price'
        )

    def _transform_data(self):
        self.data.apply_function(base_regression_pipeline)
        # Using the whole dataset. If you want to work with a sample, just uncomment the line below
        # self.data.df = self.data.df.sample(frac=0.001, replace=True, random_state=1)

    def run(self):

        # Running SVM
        svm_results: Type[Bunch] = run_svm(self)

        # Running Random Forest
        rf_results: Type[Bunch] = run_rf(self)

        # Generating combined results
        results: Type[Bunch] = Bunch(
            svm = svm_results.values(),
            rf = rf_results.values()
        )

        # Generate visualizations
        plot_actual_predicted(results, self.y_test)
        plot_residuals(results, self.y_test)
        plot_model_evaluation(results)
        plot_training_time(results)

        self.data.reset_dataset()

        return results