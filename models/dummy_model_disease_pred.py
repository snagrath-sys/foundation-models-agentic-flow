import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Global variables for reuse
encoder = LabelEncoder()
svm_model = SVC()
nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=42)
symptom_index = {}


def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    data["disease"] = encoder.fit_transform(data["disease"])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def balance_and_prepare_data(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    if 'gender' in X_resampled.columns:
        le = LabelEncoder()
        X_resampled['gender'] = le.fit_transform(X_resampled['gender'])
    X_resampled = X_resampled.fillna(0)
    if len(y_resampled.shape) > 1:
        y_resampled = y_resampled.values.ravel()
    return X_resampled, y_resampled


def evaluate_models(X, y):
    models = {
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
            print("=" * 50)
            print(f"Model: {model_name}")
            print(f"Scores: {scores}")
            print(f"Mean Accuracy: {scores.mean():.4f}")
        except Exception as e:
            print("=" * 50)
            print(f"Model: {model_name} failed with error:")
            print(e)


def train_models(X, y):
    svm_model.fit(X, y)
    nb_model.fit(X, y)
    rf_model.fit(X, y)

    return {
        "SVM": svm_model.predict(X),
        "Naive Bayes": nb_model.predict(X),
        "Random Forest": rf_model.predict(X)
    }


def visualize_confusion_matrices(y_true, predictions):
    for name, preds in predictions.items():
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt="d")
        plt.title(f"Confusion Matrix for {name} Classifier")
        plt.show()
        print(f"{name} Accuracy: {accuracy_score(y_true, preds) * 100:.2f}%")

    combined_preds = [mode([predictions['SVM'][i], predictions['Naive Bayes'][i], predictions['Random Forest'][i]]) for i in range(len(y_true))]
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_true, combined_preds), annot=True, fmt="d")
    plt.title("Confusion Matrix for Combined Model")
    plt.show()
    print(f"Combined Model Accuracy: {accuracy_score(y_true, combined_preds) * 100:.2f}%")


def build_symptom_index(X_columns):
    global symptom_index
    symptoms = X_columns.values
    symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def init_models():
    print("init model")
    csv_path = '/content/data/improved_disease_dataset.csv'
    X, y = load_and_preprocess_data(csv_path)
    X_resampled, y_resampled = balance_and_prepare_data(X, y)
    evaluate_models(X_resampled, y_resampled)
    predictions = train_models(X_resampled, y_resampled)
    visualize_confusion_matrices(y_resampled, predictions)
    build_symptom_index(X.columns)


def predict_disease(input_symptoms):
    print("predict_disease")
    init_models()
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)
    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]
    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }


if __name__ == "__main__":
    csv_path = '/content/data/improved_disease_dataset.csv'
    X, y = load_and_preprocess_data(csv_path)
    X_resampled, y_resampled = balance_and_prepare_data(X, y)
    evaluate_models(X_resampled, y_resampled)
    predictions = train_models(X_resampled, y_resampled)
    visualize_confusion_matrices(y_resampled, predictions)
    build_symptom_index(X.columns)

    print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))