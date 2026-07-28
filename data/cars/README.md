# Used Car Price Prediction

This project aims to build a machine learning model that predicts the price of a used car based on various features. The goal is to create a reliable model that accurately estimates the price in USD, taking into account key factors such as car manufacturer, condition, mileage, fuel type, and more.

## Dataset

The dataset includes the following features:

- **company**: The manufacturer of the car.
- **model**: The specific model or brand of the car.
- **year**: The year the car was manufactured.
- **condition**: The type of sale (e.g., used, for parts, etc.).
- **mileage**: The total mileage driven by the car.
- **fuel**: The type of fuel the car uses (e.g., petrol, diesel, electric).
- **volume (cm³)**: The engine size in cubic centimeters.
- **color**: The color of the car.
- **transmission**: The type of transmission (e.g., automatic, manual).
- **drive_unit**: The drive type of the car (e.g., front-wheel drive, rear-wheel drive, all-wheel drive).
- **vehicle_size_class**: A manually labeled classification of the vehicle size (note: it may contain errors).
- **price (USD)**: The target variable representing the price of the car in US dollars.

## Goals

1. **Data Preprocessing**: Handle categorical variables carefully and ensure proper encoding.
2. **Model Training**: Build a model that can predict the price of a car based on the features provided.
3. **Model Evaluation**: Ensure the model performs well on unseen data and does not produce unrealistic results (e.g., negative prices).
4. **Feature Engineering**: Explore and transform the features to improve the model's accuracy.

## Special Considerations

- **Categorical Variables**: Be extra cautious when processing categorical data such as the company, model, condition, fuel, and transmission types.
- **Model Validation**: Ensure the model is robust and performs well, particularly avoiding errors like predicting negative prices.
- **Data Quality**: Pay attention to potential data issues, especially with manually labeled columns such as vehicle size class.

