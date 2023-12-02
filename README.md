# House Price Prediction using Linear Regression

This project is a part of the Prodigy Infotech Machine Learning Internship, focusing on implementing a linear regression model to predict house prices based on various features.

## Author

- **Author:** Muhammad Awais Akhter
- **GitHub:** [awais-akhter](https://github.com/awais-akhter)

## Problem Statement

The task involves developing a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms. The dataset used for this project can be accessed via the following Kaggle link:
[Dataset Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## Implementation Details

### Libraries Used
- `numpy`, `pandas`, `matplotlib`, `seaborn`: Data manipulation and visualization
- `SimpleImputer` from `sklearn`: Handling missing data
- `StandardScaler` and `LabelEncoder` from `sklearn`: Data preprocessing
- `train_test_split` from `sklearn.model_selection`: Splitting the dataset
- Various regression models from `sklearn` and `xgboost`

### Workflow

- **Data Loading**: Loaded the training and testing datasets from Kaggle.
- **Handling Missing Data**: Identified and handled missing data in the dataset using imputation techniques.
- **Data Preprocessing**: Scaled numerical features using Standard Scaler and encoded categorical features using Label Encoder.
- **Model Training**: Trained various regression models including Random Forest, Decision Tree, Gradient Boosting, Support Vector, K Neighbors, and XG Boost.
- **Model Evaluation**: Evaluated the models using metrics such as R-squared, Mean Absolute Error, and Mean Squared Error. Identified Gradient Boosting as the best-performing model.
- **Cross-Validation**: Conducted K-Fold Cross Validation to validate the model's performance.
- **Prediction**: Used the trained Gradient Boosting model to predict house prices on the test dataset.

### Conclusion

The Gradient Boosting Regressor emerged as the best-performing model for house price prediction. The project includes comprehensive data handling, preprocessing, and model evaluation steps.

## Execution

To run this code locally, follow these steps:
1. Download the dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
2. Set up the Jupyter Notebook environment or Python with necessary libraries.
3. Execute the code cells in the provided `House_Price_Predicton_using_Regression.ipynb` file.

## Acknowledgments

- Kaggle for hosting the dataset used in this project.

Feel free to explore the code and dataset further for insights and improvements!
