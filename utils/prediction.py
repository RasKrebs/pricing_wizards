# Data Manipulation
from typing import Type

# Scikit learn
from sklearn.utils import Bunch

# Load helpers and custom dataset class
from scripts.svm import run_svm
from scripts.random_forest import run_rf
from utils.Dataloader import PricingWizardDataset
from utils.helpers import plot_actual_predicted, plot_residuals

class Prediction:
    def __init__(self,
                data: PricingWizardDataset = None):

        # Setting values
        self.data = data

        # Define indenpendent variables
        independent_variables = ['brand_name', 'category_name', 'condition_name', 'viewed_count', 'subcategory_name']

        # Get train and test data
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = data.stratify_train_test_split(independent_variables=independent_variables,
                                                                                                                      y_column='listing_price',)

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

        return results