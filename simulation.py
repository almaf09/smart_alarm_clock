import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sleep_efficiency_model import load_and_preprocess_data, create_ensemble_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

def predict_sleep_efficiency(features):
    _, _, _, _ = load_and_preprocess_data()
    ensemble_model = create_ensemble_model(len(features))
    ensemble_model.load_weights('ensemble_model_weights.h5')
    return ensemble_model.predict(np.array([features]))[0][0]

def find_optimal_wakeup_time(bedtime, desired_wakeup_start, desired_wakeup_end, features):
    bedtime_time = datetime.strptime(bedtime, '%H:%M')
    start_time = datetime.strptime(desired_wakeup_start, '%H:%M')
    end_time = datetime.strptime(desired_wakeup_end, '%H:%M')
    current_time = start_time
    best_efficiency = -1
    best_time = None

    while current_time <= end_time:
        sleep_duration = (current_time - bedtime_time).total_seconds() / 3600  # Convert to hours
        features_with_time = features + [sleep_duration, current_time.hour, current_time.minute]
        efficiency = predict_sleep_efficiency(features_with_time)
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_time = current_time
        current_time += timedelta(minutes=10)

    return best_time.strftime('%H:%M'), best_efficiency

def load_and_preprocess_data():
    # Read the dataset
    df = pd.read_csv('Sleep_Efficiency.csv')
    
    # Convert categorical variables first
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Smoking status'] = le.fit_transform(df['Smoking status'])
    
    # Handle missing values for numerical columns only
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    
    # Add Wakeup hour and Wakeup minute as new features (simulate with random values for training)
    np.random.seed(42)
    df['Wakeup hour'] = np.random.randint(0, 24, size=len(df))
    df['Wakeup minute'] = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=len(df))
    
    features = ['Age', 'Gender', 'REM sleep percentage', 'Deep sleep percentage', 'Light sleep percentage',
                'Awakenings', 'Caffeine consumption', 'Alcohol consumption', 'Smoking status', 'Exercise frequency',
                'Sleep duration', 'Wakeup hour', 'Wakeup minute']
    
    X = df[features].values
    y = df['Sleep efficiency'].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return scaler, features

def create_ensemble_model(input_dim):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Neural Network branch
    nn_branch = tf.keras.layers.Dense(64, activation='relu')(inputs)
    nn_branch = tf.keras.layers.Dropout(0.2)(nn_branch)
    nn_branch = tf.keras.layers.Dense(32, activation='relu')(nn_branch)
    nn_branch = tf.keras.layers.Dropout(0.2)(nn_branch)
    nn_branch = tf.keras.layers.Dense(16, activation='relu')(nn_branch)
    nn_branch = tf.keras.layers.Dense(1, activation='sigmoid')(nn_branch)
    
    # LSTM branch
    lstm_branch = tf.keras.layers.Reshape((1, input_dim))(inputs)
    lstm_branch = tf.keras.layers.LSTM(64, return_sequences=True)(lstm_branch)
    lstm_branch = tf.keras.layers.Dropout(0.2)(lstm_branch)
    lstm_branch = tf.keras.layers.LSTM(32)(lstm_branch)
    lstm_branch = tf.keras.layers.Dropout(0.2)(lstm_branch)
    lstm_branch = tf.keras.layers.Dense(16, activation='relu')(lstm_branch)
    lstm_branch = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_branch)
    
    # Combine branches
    combined = tf.keras.layers.Average()([nn_branch, lstm_branch])
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=combined)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # Load the preprocessed data and get the scaler
    scaler, features = load_and_preprocess_data()
    
    # Create and load the model
    model = create_ensemble_model(len(features))
    model.load_weights('ensemble_model_weights.h5')
    
    # Get user input
    bedtime = input("Enter bedtime (HH:MM): ")
    wakeup_start = input("Enter desired wakeup start time (HH:MM): ")
    wakeup_end = input("Enter desired wakeup end time (HH:MM): ")
    
    # Convert times to datetime objects
    bedtime_dt = datetime.strptime(bedtime, "%H:%M")
    wakeup_start_dt = datetime.strptime(wakeup_start, "%H:%M")
    wakeup_end_dt = datetime.strptime(wakeup_end, "%H:%M")
    
    # Get other user inputs
    age = int(input("Enter Age: "))
    gender = int(input("Enter Gender (0 for Female, 1 for Male): "))
    rem_sleep = float(input("Enter REM sleep percentage: "))
    deep_sleep = float(input("Enter Deep sleep percentage: "))
    light_sleep = float(input("Enter Light sleep percentage: "))
    awakenings = int(input("Enter Awakenings: "))
    caffeine = float(input("Enter Caffeine consumption: "))
    alcohol = float(input("Enter Alcohol consumption: "))
    smoking = int(input("Enter Smoking status (0 for No, 1 for Yes): "))
    exercise = int(input("Enter Exercise frequency: "))
    
    # Calculate sleep efficiencies for each time slot
    best_efficiency = 0
    best_time = None
    all_efficiencies = []
    
    current_time = wakeup_start_dt
    while current_time <= wakeup_end_dt:
        # Calculate sleep duration
        if current_time < bedtime_dt:
            current_time = current_time.replace(day=current_time.day + 1)
        sleep_duration = (current_time - bedtime_dt).total_seconds() / 3600  # Convert to hours
        
        # Prepare input features
        input_data = np.array([[
            age, gender, rem_sleep, deep_sleep, light_sleep,
            awakenings, caffeine, alcohol, smoking, exercise,
            sleep_duration, current_time.hour, current_time.minute
        ]])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        efficiency = model.predict(input_scaled, verbose=0)[0][0]
        
        # Store the efficiency
        all_efficiencies.append((current_time.strftime("%H:%M"), efficiency))
        
        # Update best time if current efficiency is better
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_time = current_time
        
        # Move to next time slot (5-minute intervals)
        current_time += timedelta(minutes=5)
    
    # Print all efficiencies
    print("\nAll calculated sleep efficiencies:")
    print("Time\t\tEfficiency")
    print("-" * 30)
    for time, eff in all_efficiencies:
        print(f"{time}\t\t{eff:.4f}")
    
    # Print the best time
    print(f"\nOptimal wakeup time: {best_time.strftime('%H:%M')} with sleep efficiency: {best_efficiency:.4f}")

if __name__ == "__main__":
    main()

# Allowed inputs for each feature:
# Age: Numeric value (e.g., 25)
# Gender: Numeric value (e.g., 0 for Female, 1 for Male)
# REM sleep percentage: Numeric value (e.g., 20.5)
# Deep sleep percentage: Numeric value (e.g., 15.0)
# Light sleep percentage: Numeric value (e.g., 50.0)
# Awakenings: Numeric value (e.g., 2)
# Caffeine consumption: Numeric value (e.g., 100)
# Alcohol consumption: Numeric value (e.g., 0)
# Smoking status: Numeric value (e.g., 0 for Non-smoker, 1 for Smoker)
# Exercise frequency: Numeric value (e.g., 3) 