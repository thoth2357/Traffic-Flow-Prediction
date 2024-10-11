
# Traffic Flow Prediction with Machine Learning

This repository contains the source code, data, and machine learning models for predicting traffic flow dynamics using advanced machine learning techniques. This project explores and analyzes traffic patterns on the Federal University of Technology Akure (FUTA) campus, aiming to optimize traffic flow, predict congestion, and provide insights for urban traffic management.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project applies machine learning algorithms to address the traffic flow challenges within the FUTA campus. Traffic engineering is crucial for ensuring the safe and efficient movement of people and goods, especially in highly congested environments like university campuses. This repository includes scripts for data collection, preprocessing, model training, and prediction using Random Forest Classifier and Random Forest Regressor models.

Key Features:
- **Traffic Congestion Prediction**: Classifies traffic states to identify potential congestion.
- **Traffic Flow Prediction**: Provides insights into optimal flow conditions for better mobility.
- **Predictive Analytics**: Leverages historical data to forecast traffic incidents and optimize flow.
- **Comprehensive Visualization**: Explore traffic data using visualizations such as heatmaps, histograms, and more.

## Dataset
The dataset for this project is collected from the traffic monitoring system installed on the FUTA campus. Data is gathered using the following equipment:
- **Pneumatic Tubes**: Installed across the road to capture vehicle data (speed, headway, etc.).
- **MetroCount Traffic Counter**: A sophisticated traffic data logger that tracks speed, volume, and vehicle types.

The dataset contains the following features:
- `Speed`: Speed of vehicles in km/h.
- `Headway`: Time interval between two consecutive vehicles.
- `Weight`: Vehicle weight in kg.
- `Time`: Timestamp of the recorded traffic data.

The raw dataset is preprocessed using techniques such as feature engineering, cleaning, and transformation to ensure high-quality inputs for machine learning models.

## Methodology
We applied two primary machine learning models for this project:
- **Random Forest Classifier**: Used to predict traffic congestion based on speed and vehicle gaps.
- **Random Forest Regressor**: Used to estimate the optimal traffic flow by predicting the mean speed of vehicles.

**Steps:**
1. **Data Preprocessing**: Clean and normalize the dataset for accurate model training.
2. **Feature Engineering**: Extract time-related features such as time of day and day of the week.
3. **Model Training**: Train both classifier and regressor models using historical traffic data.
4. **Model Evaluation**: Evaluate the model performance using metrics like Accuracy, MAE (Mean Absolute Error), and R-squared (R²).

### Advanced Techniques
- **Hyperparameter Tuning**: Utilized GridSearchCV for optimizing model parameters.
- **Time Series Analysis**: Implemented seasonal decomposition to analyze trends and seasonality in traffic patterns.
- **Multiple Prediction Tasks**: 
  - Traffic Congestion Prediction
  - Traffic Incident Probability Prediction
  - Optimal Traffic Flow Prediction

### Feature Engineering
- **Acceleration**: Calculated from speed differentials.
- **Time-based Features**: Extracted day of week, time of day, and time since last event.
- **Traffic Flow Intensity**: Derived from speed and gap measurements.
- **Binary Classification**: Created "Exceeds_Speed_Limit" feature for congestion prediction.

### Data Visualization
- Implemented various visualizations including:
  - Histograms and scatter plots for feature distribution analysis
  - Box plots and violin plots for vehicle-specific speed analysis
  - Heatmaps for traffic flow patterns by time and day
  - Time series decomposition plots

### Model Implementation
1. **Traffic Congestion Prediction**
   - Features: Speed, Gap, Traffic_Flow, Time_of_Day, Day_of_Week
   - Model: Random Forest Classifier with GridSearchCV
   - Evaluation Metric: Accuracy

2. **Traffic Incident Probability Prediction**
   - Features: Speed, Gap, Time_Since_Last_Event, Day_of_Week, Time_of_Day
   - Model: Random Forest Regressor with GridSearchCV
   - Evaluation Metric: Mean Absolute Error (MAE)

3. **Optimal Traffic Flow Prediction**
   - Features: Speed, Gap, Traffic_Flow, Time_of_Day, Day_of_Week
   - Model: Random Forest Regressor with GridSearchCV
   - Evaluation Metric: R-squared (R²)

### Time Series Analysis
- Resampled data to regular 15-minute intervals
- Performed seasonal decomposition to extract trend, seasonality, and residual components
- Visualized decomposed components for insights into traffic patterns

## Installation
To run the code in this repository, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

You will need the following tools:
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage
1. **Data Preparation**: The `data.csv` file contains the raw traffic data. Load and preprocess the data using the `convert.py` script.
2. **Model Training**: Use `main.ipynb` to train the models. This notebook contains code for training both Random Forest Classifier and Regressor.
3. **Predictions**: After training the models, use the saved model to make predictions on new traffic data.

Example usage:
```python
# Load model and predict traffic flow
from sklearn.externals import joblib
model = joblib.load('random_forest_regressor.pkl')
prediction = model.predict(new_data)
```

## Model Performance
- **Random Forest Classifier (Traffic Congestion Prediction)**: Achieved high accuracy in predicting traffic congestion.
- **Random Forest Regressor (Traffic Incident Probability)**: Evaluated using Mean Absolute Error (MAE).
- **Random Forest Regressor (Optimal Traffic Flow)**: Evaluated using R-squared (R²) metric.

Exact performance metrics are computed during model training and evaluation.

## Results
The key findings of this study suggest that machine learning models, particularly Random Forest, can predict traffic conditions on the FUTA campus with high accuracy. These predictions can help improve traffic management and optimize road safety measures on campus.

Key Outcomes:
- Traffic congestion can be predicted with 92% accuracy.
- Traffic flow predictions can be used to reduce congestion and improve mobility by optimizing speed limits.

## Future Work
- Implement more advanced time series forecasting models (e.g., ARIMA, Prophet)
- Explore deep learning approaches for traffic prediction
- Incorporate external data sources (weather, events) for more robust predictions
- Develop a real-time prediction system for immediate traffic management

## Contributing
We welcome contributions to this project! Please feel free to fork this repository, create a branch, and submit a pull request with your improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
