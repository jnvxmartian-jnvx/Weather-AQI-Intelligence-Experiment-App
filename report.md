# Weather and AQI Analysis of Indian Cities

## Project Objective

The objective of this project is to analyze weather conditions and air quality across Indian cities and build a machine learning model to predict Air Quality Index (AQI).

## Dataset Description

The dataset contains weather and air quality observations from multiple Indian cities.

### Features

* Date
* City
* State
* Maximum Temperature
* Minimum Temperature
* Average Temperature
* Humidity
* Rainfall
* Wind Speed
* AQI
* AQI Category
* Pressure
* Visibility

### Total Records

599 records

## Tools and Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* VS Code

## Data Cleaning

The following preprocessing steps were performed:

* Removed duplicate records
* Converted numerical columns to proper data types
* Removed invalid and incomplete rows
* Checked for missing values

## Exploratory Data Analysis

The following visualizations were created:

1. Average Temperature by City
2. AQI Distribution
3. Rainfall vs AQI Scatter Plot
4. Correlation Heatmap
5. Feature Importance Graph

## Machine Learning Model

Model Used:

* Random Forest Regressor

Input Features:

* Average Temperature
* Humidity
* Rainfall
* Wind Speed
* Pressure
* Visibility

Target Variable:

* AQI

## Results

The model was trained and evaluated using a train-test split.

Performance metrics:

* Mean Absolute Error (MAE): To be added after execution
* R² Score: To be added after execution

## Conclusion

Weather conditions influence air quality and can be used to estimate AQI values. Machine learning techniques provide an effective method for predicting AQI based on environmental factors.

## Future Improvements

* Larger dataset
* More cities
* Streamlit dashboard
* Real-time weather API integration
* Advanced machine learning models
