import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import tkinter as tk
from tkinter import ttk, messagebox

warnings.filterwarnings("ignore")
np.random.seed(40)

class ElasticNetRegression:
    def __init__(self, alpha=0.5, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def train_neural_network(X, y, epochs=1000, lr=0.01):
    input_size = X.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = torch.tensor(X.values, dtype=torch.float32)
        targets = torch.tensor(y.values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)
    predicted_qualities = model.predict(test_x)
    return eval_metrics(test_y, predicted_qualities)

def train_and_log(model, alpha, l1_ratio, X, y):
    with mlflow.start_run():
        model.fit(X, y)
        predicted_qualities = model.predict(X)
        (rmse, mae, r2) = eval_metrics(y, predicted_qualities)

        print("Model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = model.predict(X)
        signature = infer_signature(X, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

def create_gui(model):
    def on_predict():
        try:
            alpha_val = float(alpha_entry.get())
            l1_ratio_val = float(l1_ratio_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            return

        predictions = model.predict(np.array([[alpha_val, l1_ratio_val]]))
        result_label.config(text=f"Predicted Quality: {predictions[0]:.2f}")

    root = tk.Tk()
    root.title("Wine Quality Predictor")

    alpha_label = ttk.Label(root, text="Alpha:")
    alpha_label.grid(row=0, column=0, padx=10, pady=10)
    alpha_entry = ttk.Entry(root)
    alpha_entry.grid(row=0, column=1, padx=10, pady=10)

    l1_ratio_label = ttk.Label(root, text="L1 Ratio:")
    l1_ratio_label.grid(row=1, column=0, padx=10, pady=10)
    l1_ratio_entry = ttk.Entry(root)
    l1_ratio_entry.grid(row=1, column=1, padx=10, pady=10)

    predict_button = ttk.Button(root, text="Predict", command=on_predict)
    predict_button.grid(row=2, column=0, columnspan=2, pady=10)

    result_label = ttk.Label(root, text="")
    result_label.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    # Read the wine-quality csv file from the URL
    csv_url = (
        "
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Train and evaluate ElasticNet model
    elasticnet_model = ElasticNetRegression()
    elasticnet_metrics = train_and_evaluate(elasticnet_model, train_x, train_y, test_x, test_y)

    print("Elasticnet model:")
    print("  RMSE: %s" % elasticnet_metrics[0])
    print("  MAE: %s" % elasticnet_metrics[1])
    print("  R2: %s" % elasticnet_metrics[2])

    # Train and evaluate Neural Network model
    neural_network_model = train_neural_network(train_x, train_y)
    neural_network_predictions = neural_network_model(torch.tensor(test_x.values, dtype=torch.float32)).detach().numpy()
    neural_network_metrics = eval_metrics(test_y, neural_network_predictions)

    print("Neural Network model:")
    print("  RMSE: %s" % neural_network_metrics[0])
    print("  MAE: %s" % neural_network_metrics[1])
    print("  R2: %s" % neural_network_metrics[2])

    # Log ElasticNet model in MLflow
    train_and_log(elasticnet_model, 0.5, 0.5, train_x, train_y)

    # Create GUI for model predictions
    create_gui(elasticnet_model)
