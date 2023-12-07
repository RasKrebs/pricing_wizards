from sklearn.model_selection import train_test_split

from models.svm import run_model as run_svm
from models.random_forest import run_model as run_rf
from feature_importance import rank_feature_importances

def run_predictions(model_config):
    # Running SVM prediction
    svm_results: dict = run_svm(model_config)

    # Running Random Forest prediction
    rf_results: dict = run_rf(model_config)

    # Extract only feature importances from the results
    feature_importances: dict = {
        'SVR Linear': svm_results['SVR Linear']['feature_importances'],
        'Random Forest': rf_results['Random Forest']['feature_importances']
    }

    # Rank feature importances based on the values
    # ranked_feature_importances = rank_feature_importances(feature_importances)

