# Sleep Efficiency Prediction System

This project implements a deep learning system to predict optimal sleep efficiency and wake-up times based on various sleep-related parameters. The system uses an ensemble of neural networks to provide accurate predictions and recommendations.

## Dataset

The project uses the Sleep Efficiency Dataset, which can be found at:
[Sleep Efficiency Dataset on Kaggle](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep)

The dataset contains information about sleep patterns and efficiency, including:
- Sleep duration
- Sleep efficiency
- REM, Deep, and Light sleep percentages
- Awakenings
- Lifestyle factors
- Demographic information

## Features

- Sleep efficiency prediction using an ensemble of neural networks
- Optimal wake-up time calculation within a specified time window
- Support for various sleep parameters including:
  - Sleep duration
  - REM, Deep, and Light sleep percentages
  - Awakenings
  - Lifestyle factors (caffeine, alcohol, smoking, exercise)
  - Age and gender
- Visualization of training history and feature correlations
- Interactive simulation interface

## Requirements

Required Python packages with specific versions:
```
tensorflow==2.15.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
```

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure and File Descriptions

### Core Files
- `sleep_efficiency_model.py`: 
  - Contains the model architecture and training code
  - Implements data preprocessing, model creation, and training functions
  - Generates training visualizations and correlation matrix
  - Saves the trained model weights

- `simulation.py`: 
  - Interactive script for sleep efficiency prediction
  - Takes user input for sleep parameters
  - Calculates and displays sleep efficiency for different wake-up times
  - Recommends optimal wake-up time

### Data and Model Files
- `Sleep_Efficiency.csv`: 
  - Raw dataset containing sleep-related parameters
  - Required for training the model
  - Should be placed in the project root directory

- `ensemble_model_weights.h5`: 
  - Saved weights of the trained ensemble model
  - Generated after training
  - Required for running the simulation

### Output Files
- `training_history.png`: 
  - Visualization of model training progress
  - Shows loss and validation metrics over epochs
  - Generated during model training

- `correlation_matrix.png`: 
  - Heatmap showing correlations between features
  - Helps understand feature relationships
  - Generated during model training

## Usage

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd sleep-efficiency-prediction
```

2. Download the dataset:
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep)
   - Download `Sleep_Efficiency.csv`
   - Place it in the project root directory

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

To train the model:
```bash
python3 sleep_efficiency_model.py
```

This will:
1. Train the neural network models
2. Generate training history plots
3. Create a correlation matrix visualization
4. Save the model weights

### Running the Simulation

To run the sleep efficiency simulation:
```bash
python3 simulation.py
```

The simulation will prompt you for:
1. Bedtime (HH:MM)
2. Desired wake-up time window (start and end times)
3. Sleep parameters:
   - Age
   - Gender (0 for Female, 1 for Male)
   - REM sleep percentage
   - Deep sleep percentage
   - Light sleep percentage
   - Number of awakenings
   - Caffeine consumption
   - Alcohol consumption
   - Smoking status (0 for No, 1 for Yes)
   - Exercise frequency

The simulation will then:
1. Calculate sleep efficiency for each 5-minute interval in your desired wake-up window
2. Display all calculated efficiencies
3. Recommend the optimal wake-up time

## Model Architecture

The system uses an ensemble model combining:
1. A Neural Network branch
2. An LSTM branch

Both branches use:
- Multiple dense layers with ReLU activation
- Dropout layers for regularization
- Sigmoid activation for the output layer

The ensemble combines predictions from both branches using averaging.

## Performance

The model achieves:
- Mean Squared Error (MSE): ~0.0028
- R² Score: ~0.85

## Troubleshooting

Common issues and solutions:

1. **Model weights not found**:
   - Ensure you've run `sleep_efficiency_model.py` first
   - Check if `ensemble_model_weights.h5` exists in the project directory

2. **Dataset not found**:
   - Verify `Sleep_Efficiency.csv` is in the project root directory
   - Check if the file name matches exactly

3. **Package version conflicts**:
   - Use the exact versions specified in requirements.txt
   - Create a virtual environment for isolation

4. **Memory issues**:
   - Reduce batch size in `sleep_efficiency_model.py`
   - Close other memory-intensive applications

## License

This project is open source and available under the MIT License.

