import os
import pickle
from pathlib import Path
import numpy as np

# current_path = Path.cwd()
script_path = Path(__file__).resolve()
models_directory = script_path.parent.parent.parent.parent / "savedModels" / "model"
print(models_directory)
most_recent_model = max(models_directory.glob("*"), key=lambda f: f.stat().st_mtime)

def load_model():
    print("Most Recent Model:", most_recent_model)
    try:
        model = pickle.load(open(most_recent_model, 'rb'))
        return model
    except FileNotFoundError:
        print("The pickle file does not exist.")
    except pickle.UnpicklingError:
        print("The pickle file could not be loaded.")

def train(new_X, new_y):
    classes = np.unique(new_y)
    SGD_model = load_model()

    SGD_model.partial_fit(new_X, new_y, classes=classes)
    print(most_recent_model.stem)
    # pickle.dump(SGD_model, open('updated_model.sav', 'wb'))

# def predict()
