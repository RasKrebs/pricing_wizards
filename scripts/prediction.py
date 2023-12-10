from typing import Type
from sklearn.utils import Bunch

from scripts.svm import main as run_svm
from scripts.random_forest import main as run_rf
from scripts.feature_importance import rank_feature_importances
from utils.object import MLModelConfig

def main(model_config: MLModelConfig) -> Type[Bunch]:
    """
    Run predictions using the provided model configuration.

    Parameters:
    - model_config (MLModelConfig): Configuration object for the machine learning models.

    Returns:
    - Type[Bunch]: A Bunch object containing the results of predictions.
    """

    # Running SVM prediction
    svm_results: Type[Bunch] = run_svm(model_config)

    # Running Random Forest prediction
    rf_results: Type[Bunch] = run_rf(model_config)

    # Generating combined results to return
    results: Type[Bunch] = Bunch(
        svm = svm_results.values(),
        rf = rf_results.values()
    )

    return results

