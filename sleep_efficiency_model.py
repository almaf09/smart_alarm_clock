import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, Reshape, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the data
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
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Create Neural Network model
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create LSTM model
def create_lstm_model(input_dim):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, input_dim)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create ensemble model
def create_ensemble_model(input_dim):
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Neural Network branch
    nn_branch = Dense(64, activation='relu')(inputs)
    nn_branch = Dropout(0.2)(nn_branch)
    nn_branch = Dense(32, activation='relu')(nn_branch)
    nn_branch = Dropout(0.2)(nn_branch)
    nn_branch = Dense(16, activation='relu')(nn_branch)
    nn_branch = Dense(1, activation='sigmoid')(nn_branch)
    
    # LSTM branch
    lstm_branch = Reshape((1, input_dim))(inputs)
    lstm_branch = LSTM(64, return_sequences=True)(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = LSTM(32)(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Dense(16, activation='relu')(lstm_branch)
    lstm_branch = Dense(1, activation='sigmoid')(lstm_branch)
    
    # Combine branches
    combined = Average()([nn_branch, lstm_branch])
    
    # Create model
    model = Model(inputs=inputs, outputs=combined)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_models():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train individual models
    nn_model = create_nn_model(X_train.shape[1])
    lstm_model = create_lstm_model(X_train.shape[1])
    ensemble_model = create_ensemble_model(X_train.shape[1])
    
    # Train models with early stopping
    print("Training Neural Network model...")
    nn_history = nn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nTraining LSTM model...")
    lstm_history = lstm_model.fit(
        X_train.reshape(-1, 1, X_train.shape[1]),
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nTraining Ensemble model...")
    ensemble_history = ensemble_model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save ensemble model weights
    ensemble_model.save_weights('ensemble_model_weights.h5')
    
    # Evaluate models
    nn_pred = nn_model.predict(X_test)
    lstm_pred = lstm_model.predict(X_test.reshape(-1, 1, X_test.shape[1]))
    ensemble_pred = ensemble_model.predict(X_test)
    
    # Calculate metrics
    models = {
        'Neural Network': nn_pred,
        'LSTM': lstm_pred,
        'Ensemble': ensemble_pred
    }
    
    for name, predictions in models.items():
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"\n{name} Model Results:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(lstm_history.history['loss'], label='Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(ensemble_history.history['loss'], label='Training Loss')
    plt.plot(ensemble_history.history['val_loss'], label='Validation Loss')
    plt.title('Ensemble Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def visualize_correlation_matrix():
    df = pd.read_csv('Sleep_Efficiency.csv')
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Smoking status'] = le.fit_transform(df['Smoking status'])
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    features = ['Age', 'Gender', 'Sleep duration', 'REM sleep percentage', 'Deep sleep percentage', 'Light sleep percentage', 'Awakenings', 'Caffeine consumption', 'Alcohol consumption', 'Smoking status', 'Exercise frequency']
    corr_matrix = df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

if __name__ == "__main__":
    train_and_evaluate_models()
    visualize_correlation_matrix() 