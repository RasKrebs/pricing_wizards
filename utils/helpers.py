import pickle

def save_model(dictionary, path):
    model = dictionary['model']
    if model is not None:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
            print(f"Model saved successfully at {path}")
    else:
        print("No model found in the dictionary.")
