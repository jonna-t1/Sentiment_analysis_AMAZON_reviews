import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

# current_path = Path.cwd()
script_path = Path(__file__).resolve()
models_directory = script_path.parent.parent.parent.parent / "savedModels" / "model"
print(models_directory)
most_recent_model = max(models_directory.glob("*"), key=lambda f: f.stat().st_mtime)

def remove_last_two_if_not_letters(s):
    # Check if the last two characters are not letters
    if len(s) >= 2 and not (s[-1].isalpha() and s[-2].isalpha()):
        return s[:-2]  # Remove the last two characters
    return s  # Return the original string if the condition is not met

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
    most_recent_stem = most_recent_model.stem
    # Check for existing numbered files and determine the next number
    existing_numbers = [
        int(file.stem.rsplit("_", 1)[-1])
        for file in models_directory.glob("*")
        if "_" in file.stem and file.stem.rsplit("_", 1)[-1].isdigit()
    ]
    # Get the next number for the new file
    next_number = max(existing_numbers, default=0) + 1
    suff = ".sav"
    result = remove_last_two_if_not_letters(most_recent_stem)

    # Create the new filename by appending the next number
    new_model_filename = f"{result}_{next_number}{suff}"
    print(new_model_filename)
    retrained_model = models_directory / new_model_filename
    print(retrained_model)
    try:
        pickle.dump(SGD_model, open(retrained_model, 'wb'))
        print(f"Model saved successfully to {retrained_model}")
    except FileNotFoundError:
        print(f"Error: The file path '{retrained_model}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{retrained_model}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def predict(X_test, y_true):
    print("Predicting...")
    new_model = max(models_directory.glob("*"), key=lambda f: f.stat().st_mtime)
    model = pickle.load(open(new_model, 'rb'))
    y_pred = model.predict(X_test)
    report = classification_report(y_true, y_pred, output_dict=True)
    return y_pred, report