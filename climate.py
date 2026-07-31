import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Global variables
data = None
years = None
temp = None
model = None
future_years = None
future_temp = None

# ---------------- LOAD DATA ---------------- #
def load_data():
    global data, years, temp

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        return

    data = pd.read_csv(file_path)

    # 👉 Adjust column names if needed
    # Common Kaggle dataset columns:
    # 'dt', 'LandAverageTemperature'

    if 'Year' in data.columns:
        years = data['Year'].values.reshape(-1,1)
    else:
        data['Year'] = pd.to_datetime(data['dt']).dt.year
        years = data['Year'].values.reshape(-1,1)

    if 'LandAverageTemperature' in data.columns:
        temp = data['LandAverageTemperature'].values
    else:
        temp = data.iloc[:,1].values  # fallback

    # Remove missing values
    mask = ~np.isnan(temp)
    years = years[mask]
    temp = temp[mask]

    status_label.config(text="Data Loaded Successfully")

# ---------------- TRAIN MODEL ---------------- NOMX#
def train_model():
    global model

    if years is None:
        status_label.config(text="Load data first!")
        return

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(years)

    model = LinearRegression()
    model.fit(X_poly, temp)

    status_label.config(text="Model Trained")

# ---------------- PREDICT ---------------- #
def predict_future():
    global future_years, future_temp

    if model is None:
        status_label.config(text="Train model first!")
        return

    future_years = np.arange(2020, 2051, 5).reshape(-1,1)

    poly = PolynomialFeatures(degree=2)
    future_poly = poly.fit_transform(future_years)

    future_temp = model.predict(future_poly)

    status_label.config(text="Prediction Done")

# ---------------- GRAPH ---------------- #
def show_graph():
    if future_temp is None:
        status_label.config(text="Run prediction first!")
        return

    plt.figure()
    plt.plot(years, temp, label="Past Data")
    plt.plot(future_years, future_temp, linestyle='--', label="Predicted")

    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.title("Climate Change Prediction")
    plt.legend()
    plt.grid()

    plt.show()

# ---------------- GUI ---------------- #
root = Tk()
root.title("Climate Change Prediction System")
root.geometry("450x350")

title = Label(root, text="Climate Prediction using Numerical Methods", font=("Arial", 12, "bold"))
title.pack(pady=10)

btn_load = Button(root, text="Load Dataset (CSV)", command=load_data, width=25)
btn_load.pack(pady=5)

btn_train = Button(root, text="Train Model", command=train_model, width=25)
btn_train.pack(pady=5)

btn_predict = Button(root, text="Predict Future", command=predict_future, width=25)
btn_predict.pack(pady=5)

btn_graph = Button(root, text="Show Graph", command=show_graph, width=25)
btn_graph.pack(pady=5)

status_label = Label(root, text="Status: Waiting...", fg="blue")
status_label.pack(pady=20)

root.mainloop()