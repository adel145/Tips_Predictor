import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Function to predict tips
def predict_tips():
    try:
        # Validate shift input
        shift_input = shift_entry.get()
        if shift_input not in ["0", "1", "2"]:
            messagebox.showerror("Error", "Shift must be 0 (Morning), 1 (Evening), or 2 (Double)")
            return
        shift_input = int(shift_input)

        # Extract date and split into features
        selected_date = cal.get_date()
        date_object = datetime.strptime(selected_date, "%m/%d/%y")
        day_of_week = date_object.weekday()
        month = date_object.month
        year = date_object.year
        day = date_object.day

        # Combine input features
        input_features = np.array([[day_of_week, month, year, day, shift_input]])
        input_scaled = scaler.transform(input_features)

        # Predict tip
        prediction = model.predict(input_scaled)[0]
        messagebox.showinfo("Predicted Tip", f"Predicted Tip Value: {prediction:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Load and preprocess data
data = pd.read_csv("tips_data.csv")

# Data preparation
data['date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day

# Encode 'shift' column
label_encoder = LabelEncoder()
data['shift'] = label_encoder.fit_transform(data['shift'])

# Features and target
X = data[['day_of_week', 'month', 'year', 'day', 'shift']]
y = data['tip_value']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# GUI for input
top = tk.Tk()
top.title("Tip Prediction")
top.geometry("400x300")

# Calendar for date selection
tk.Label(top, text="Select Date:").pack(pady=5)
cal = Calendar(top, date_pattern="mm/dd/yy")
cal.pack(pady=5)

# Shift selection
tk.Label(top, text="Enter Shift (0-Morning, 1-Evening, 2-Double):").pack(pady=5)
shift_entry = tk.Entry(top)
shift_entry.pack(pady=5)

# Predict button
tk.Button(top, text="Predict Tip", command=predict_tips).pack(pady=20)

# Run the GUI
top.mainloop()
