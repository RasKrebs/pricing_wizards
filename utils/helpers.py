import pickle

def save_model(dictionary, path):
    """Saves model to the given path"""
    model = dictionary['model']
    if model is not None:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
            print(f"Model saved successfully at {path}")
    else:
        print("No model found in the dictionary.")

def load_model(path): 
    """Loads model from the given path"""
    with open(path, 'rb') as file:
        model = pickle.load(file)
        print(f"Model loaded successfully from {path}")
        return model

drop_helpers = lambda x: x.loc[:, (x.columns != 'classified_id') & (x.columns != 'listing_price')] 