## Car Price Prediction

A machine learning regression model to predict car prices using Random Forest algorithm with comprehensive feature engineering and preprocessing.

---

## Overview

This project builds a Random Forest Regression model to predict the price of cars using an extensive set of vehicle features. The notebook (`nelson_car_prediction.ipynb`) implements a complete machine learning pipeline including advanced data preprocessing, feature engineering, model training, evaluation, and persistence.

---

## Project Structure

- `nelson_car_prediction.ipynb` - Main Jupyter notebook containing the complete ML workflow
- `car_price_prediction.csv` - Dataset containing car features and prices
- `car_price_model.pkl` - Saved trained model (generated after running notebook)
- `requirements.txt` - Python dependencies
- `Pipfile` - Pipenv environment configuration

---

## Code Workflow

### 1. **Libraries Import**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
```
Imports libraries for data manipulation, visualization, machine learning, and model persistence.

### 2. **Data Loading & Initial Cleaning**
```python
car_df = pd.read_csv("car_price_prediction.csv")
car_df = car_df.replace('-', pd.NA)
car_df = car_df.dropna()
```
Loads the dataset and removes rows with missing values (including dash placeholders).

### 3. **Data Inspection**
- `car_df.head()` - View first few rows
- `car_df.columns` - Display column names
- `car_df.info()` - Show data types and structure

### 4. **Data Preprocessing**

#### Column Renaming
Renames columns to Python-friendly lowercase names with underscores for all car attributes.

#### Data Type Conversion
- Converts `levy` to float (replacing '-' with '0')
- Converts `mileage` to integer (removing 'km' suffix)
- Converts `engine_volume` to float (removing 'Turbo' suffix)
- Converts `doors` to integer with error handling

### 5. **Exploratory Data Analysis (EDA) - Visualizations**

Creates scatter plots to understand relationships:
- **Price vs Levy**: Tax/insurance cost impact
- **Price vs Production Year**: Age effect on pricing
- **Price vs Mileage**: Usage impact on depreciation
- **Price vs Cylinders**: Engine power correlation
- **Price vs Airbags**: Safety features relationship

### 6. **Feature Selection**
```python
feature_cols = ['production_year', 'levy', 'mileage', 'cylinders', 'airbags', 'doors',
                'manufacturer', 'model', 'fuel_type', 'category', 'leather_interior',
                'gear_box_type', 'drive_wheels', 'wheel', 'color', 'engine_volume']
```

### 7. **Feature Engineering**

Creates derived features to improve model performance:
- **car_age**: Current year minus production year
- **age_group**: Categorized age ranges (New, Recent, Mid-age, Old)
- **mileage_group**: Categorized mileage ranges (Low, Medium, High, Very High)
- **engine_per_cylinder**: Engine volume divided by cylinder count
- **production_year_squared**: Non-linear production year effects

### 8. **Model Building & Training**

#### Preprocessing Pipeline
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_enhanced),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols_enhanced)
    ])
```

#### Random Forest Model
```python
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=1000,
        max_depth=60,
        random_state=42,
        n_jobs=-1
    ))
])
```

#### Train-Test Split
Splits data into 75% training and 25% testing sets.

### 9. **Model Evaluation**

#### Performance Metrics
- **R² Score**: Variance explained (train vs test for overfitting check)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error (in dollars)
- **MAE**: Mean Absolute Error

#### Feature Importance Analysis
Shows which features contribute most to predictions.

### 10. **Model Persistence**
```python
joblib.dump(model, 'car_price_model.pkl')  # Save model
loaded_model = joblib.load('car_price_model.pkl')  # Load model
```

### 11. **Sample Predictions**
Demonstrates model usage with sample car data and shows predicted prices.

---

## Features Used

### Original Features
| Feature | Type | Description |
|---------|------|-------------|
| `production_year` | Numeric | Year manufactured |
| `levy` | Numeric | Tax/insurance cost |
| `mileage` | Numeric | Kilometers driven |
| `cylinders` | Numeric | Engine cylinders |
| `airbags` | Numeric | Number of airbags |
| `doors` | Numeric | Number of doors |
| `engine_volume` | Numeric | Engine displacement (L) |
| `manufacturer` | Categorical | Car brand |
| `model` | Categorical | Specific model |
| `fuel_type` | Categorical | Fuel type |
| `category` | Categorical | Car category |
| `leather_interior` | Categorical | Leather interior (Yes/No) |
| `gear_box_type` | Categorical | Transmission type |
| `drive_wheels` | Categorical | Drivetrain |
| `wheel` | Categorical | Steering side |
| `color` | Categorical | Exterior color |

### Engineered Features
| Feature | Type | Description |
|---------|------|-------------|
| `car_age` | Numeric | Years since manufacture |
| `age_group` | Categorical | Age categories |
| `mileage_group` | Categorical | Mileage categories |
| `engine_per_cylinder` | Numeric | Volume per cylinder |
| `production_year_squared` | Numeric | Non-linear age effects |

---

## Model Details

**Algorithm**: Random Forest Regressor
**Type**: Supervised Learning - Regression
**Libraries**: scikit-learn

**Hyperparameters**:
- `n_estimators`: 1000 (number of trees)
- `max_depth`: 60 (maximum tree depth)
- `random_state`: 42 (reproducibility)

**Preprocessing**:
- **Numeric features**: Standard scaling (mean=0, std=1)
- **Categorical features**: One-hot encoding with drop_first to avoid multicollinearity

---

## Performance Metrics Explained

- **R² Score**: Proportion of price variance explained (0-1). Higher = better fit.
- **MSE**: Average squared prediction errors. Lower = better accuracy.
- **RMSE**: Square root of MSE in dollars. More interpretable error metric.
- **MAE**: Average absolute prediction errors in dollars. Less sensitive to outliers.

---

## Model Persistence

The trained model is saved as `car_price_model.pkl` using joblib, allowing:
- **Deployment**: Load model without retraining
- **Inference**: Make predictions on new data
- **Sharing**: Distribute trained model to others

---

## Usage Example

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('car_price_model.pkl')

# Prepare new car data (with same preprocessing)
new_car = pd.DataFrame({...})  # Car features

# Get price prediction
predicted_price = model.predict(new_car)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

---

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- joblib

---

## Installation

### Using Pipenv (Recommended)

1. **Install Pipenv** (if not already installed):
   ```bash
   pip install pipenv
   ```

2. **Install dependencies from Pipfile**:
   ```bash
   pipenv install
   ```

3. **Activate the virtual environment**:
   ```bash
   pipenv shell
   ```

4. **Run the notebook**:
   ```bash
   jupyter notebook nelson_car_prediction.ipynb
   ```

### Using pip and requirements.txt

Alternatively, install using pip directly:
```bash
pip install -r requirements.txt
jupyter notebook nelson_car_prediction.ipynb
```


--- 