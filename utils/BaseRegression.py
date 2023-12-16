# Load helpers and custom dataset class
from utils.Dataloader import PricingWizardDataset
from utils.DataTransformation import base_regression_pipeline
from utils.helpers import save_model
from models.base_linear_regression import linear_regression

class BaseRegression:

    def __init__(self,
                 data: PricingWizardDataset = None):

        # Setting values
        self.data = data

    def run(self):
        self.data.apply_function(base_regression_pipeline)

        # Split data
        self.data.stratify_train_test_split(y_column='log_listing_price',
                                    val_size=0,
                                    return_splits=False)

        # Run model
        results = linear_regression(self.data)
        print('R2 Score:', results['r2'])
        print('MSE:', results['mse'])
        print('MAE', results['mae'])
        print('RMSE', results['rmse'])

        # Save model
        path = 'models/pickled_models/base_regression.pkl'
        save_model(results, path)

        self.data.reset_dataset()