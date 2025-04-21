import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smolagents import tool
import pandas as pd
from models.dummy_model_disease_pred import predict_disease

# Load supported symptoms from dataset column names
DATASET_PATH = "../data/improved_disease_dataset.csv"
SYMPTOM_COLUMNS = pd.read_csv(DATASET_PATH).columns[:-1].tolist()

@tool
def symptom_disease_predict(symptoms: list) -> str:
    """
    Predicts the disease based on a list of symptoms using ensemble model predictions.
    
    Args:
        symptoms: A list of symptom names to use for prediction.
        
    Returns:
        The final disease prediction from majority vote of classifiers.
        Returns a message if the prediction is inconclusive.
    """

    # Convert list to comma-separated string
    symptom_string = ",".join(symptoms)

    # Make prediction using the loaded model
    try:
        print("Tool symptom_disease_predict")
        print(symptoms)
        print(symptom_string)
        prediction_output = predict_disease(symptom_string)
        rf = prediction_output["Random Forest Prediction"]
        nb = prediction_output["Naive Bayes Prediction"]
        svm = prediction_output["SVM Prediction"]
        final = prediction_output["Final Prediction"]

        if rf == nb == svm:
            return f"The models unanimously predict: {final}"
        elif rf == final or nb == final or svm == final:
            return f"Final predicted disease based on majority: {final}"
        else:
            return (
                "The models could not conclusively determine the disease.\n"
                f"Random Forest: {rf}, Naive Bayes: {nb}, SVM: {svm}"
            )
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# print(symptom_disease_predict(["Itching", "Skin Rash", "Nodal Skin Eruptions"]))

