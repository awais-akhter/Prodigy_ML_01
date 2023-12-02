# -*- coding: utf-8 -*-
"""House_Price_Predicton_using_Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aDueMMpJ20amIuwXpDUKW3KErastgUwO

# **Prodigy Infotech - Machine Learning Internship**
### **TASK 3 - House Price Prediction using LR**

### Author : Muhammad Awais Akhter

[![alt text](https://logoeps.com/wp-content/uploads/2014/02/25231-github-cat-in-a-circle-icon-vector-icon-vector-eps.png "Git Hub Link")](https://github.com/awais-akhter)

### Problem Statement: Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

#### Dataset link :- https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
"""

# Importing Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""### Training Data"""

house_df = pd.read_csv("PRODIGY_ML_01/train.csv")
house_df

"""### Testing Data"""

test_data = pd.read_csv("PRODIGY_ML_01/test.csv")
test_data

"""### Description of the dataset"""

house_df.describe()

# Setting the Id column as the index
house_df = house_df.set_index("Id")
test_data = test_data.set_index("Id")
house_df.shape , test_data.shape

"""The training data contains 80 features and 1460 records.

The testing data contains 79 features and 1459 records.

The one extra feature in the training data is the "SalePrice" column which contains the house price we are trying to predict.

## Handling Missing Data
"""

missing_list = house_df.isna().sum().sort_values(ascending = False).head(n = 50)
missing_list

"""This data has a lot of missing data that has to be handled."""

# Fraction of data missing in each column
percentage_missing = missing_list/house_df.shape[0]
percentage_missing

# Setting a thresold to drop some selected features which have many missing values
cols_to_drop = percentage_missing[percentage_missing > 0.45].keys()
cols_to_drop

house_df.drop(cols_to_drop, axis = 1, inplace=True)
test_data.drop(cols_to_drop, axis = 1, inplace=True)

house_df.isna().sum().sort_values(ascending = False)

"""The remaining missing values will be imputed using SimpleImputer of scikit-learn."""

from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values= np.nan , strategy= 'mean') # Uses mean to impute
most_frequent_imputer = SimpleImputer(missing_values= np.nan, strategy='most_frequent') # Uses mode to impute
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median') # Uses median to impute

sns.displot(data = house_df, x = "LotFrontage")
plt.show()

"""The LotFrontage Feature seems to be almost normally distributed so we can use the mean imputer to impute it's missing values."""

house_df["LotFrontage"]=mean_imputer.fit_transform(house_df[["LotFrontage"]])
test_data["LotFrontage"]=mean_imputer.fit_transform(test_data[["LotFrontage"]])

house_df.isna().sum().sort_values(ascending = False).head(n=30)

"""There are many categorical values to be imputed. We will use the mode stategy for them."""

categorical_missing = ["GarageCond", "GarageFinish", "GarageQual","GarageType",
"BsmtExposure","BsmtCond","BsmtQual","BsmtFinType2","BsmtFinType1"]

house_df[categorical_missing] = most_frequent_imputer.fit_transform(house_df[categorical_missing])
test_data[categorical_missing] = most_frequent_imputer.fit_transform(test_data[categorical_missing])

sns.displot(data = house_df, x="GarageYrBlt")
plt.show()

"""Finally, the GarageYrBit feature is skewed to the right. So, we use meadin to impute it's missing values."""

house_df["GarageYrBlt"] = median_imputer.fit_transform(house_df[["GarageYrBlt"]])
test_data["GarageYrBlt"] = median_imputer.fit_transform(test_data[["GarageYrBlt"]])

house_df.isna().sum().sort_values(ascending = False).head(n=30)

"""Now, not much data is missing. So, we can just remove these missing entries."""

house_df.dropna(inplace = True)
test_data.dropna(inplace = True)

print(house_df.isna().sum().sort_values(ascending = False))
print(house_df.shape)
print(test_data.shape)

"""#### All missing values have been handled!"""

house_df.head()

"""## Data Preprocessing


"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

numerical_cols = house_df.select_dtypes(include = ['int64', 'float64']).columns.to_list()
numerical_cols

categorical_cols = house_df.select_dtypes(include = ['object']).columns.to_list()
categorical_cols

# Seperating features and values in the training data
X = house_df.drop('SalePrice', axis = 1)
y = house_df[['SalePrice']]
X,y

X.shape , y.shape

"""### Scaling the data using Standard Scaler"""

std_sc = StandardScaler()

def MyStandardScaler(df, col_names):
    features = df[col_names]
    std_sc.fit(features.values)
    features = std_sc.transform(features.values)
    df[col_names] = features
    return df

numerical_features = numerical_cols.copy()
numerical_features.remove('SalePrice')
X = MyStandardScaler(X , numerical_features)
test_data = MyStandardScaler(test_data, numerical_features)
std_sc.fit(y[['SalePrice']])
y = std_sc.transform(y[['SalePrice']])
y = y.reshape(-1)
X,y

"""### Encoding the categorical data in numerical form using Label Encoder"""

le = LabelEncoder()

for column in categorical_cols:
    X[column]=le.fit_transform(X[column])
    test_data[column]=le.fit_transform(test_data[column])

X.shape , test_data.shape

"""### Splitting the data in training and testing sets"""

X_train , X_test, y_train , y_test = train_test_split(X, y , test_size= 0.2, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape , y_test.shape)
print(test_data.shape)

"""## Model Training"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install xgboost
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Different models we will train
rf = RandomForestRegressor()
dt = DecisionTreeRegressor()
gb = GradientBoostingRegressor()
svr = SVR(C=100, gamma=1, kernel='linear')
kn = KNeighborsRegressor()
xg = xgb.XGBRegressor(random_state=42)

models = {"Random Forest Regression":rf, "Decision Tree Regression":dt, "Gradient Boosting":gb, "Support Vector":svr, "K Neighbors":kn, "XG Boost":xg}

model_scores = []
model_names = []
model_maes = []
model_r2scores = []

for name , model in models.items():
    model.fit(X_train , y_train)
    pred = model.predict(X_test)
    model_names.append(name)
    model_scores.append(model.score(X_test, y_test))
    model_maes.append(mean_absolute_error(pred , y_test))
    model_r2scores.append(r2_score(pred, y_test))

result_data = {
    'Model': model_names,
    'Score': model_scores,
    'MAE': model_maes,
    'R2': model_r2scores
    }

result_df = pd.DataFrame(result_data)
result_df

"""As we can see, the model which gave the best score is Gradient Boosting Regressor. So, we will use it to get the final predictions.

### Doing K Fold Cross Validation to get the model's cross validation score
"""

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = gb

X.shape, y.shape

mae_scores = []
mse_scores = []
r2_scores = []
for train_index, val_index in kfold.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)
    r2 = r2_score(y_val, y_pred)
    r2_scores.append(r2)

average_mae = np.mean(mae_scores)
average_mse = np.mean(mse_scores)
average_r2 = np.mean(r2_scores)

average_mae, average_mse, average_r2

"""### Using the model to find Predictions"""

predictions = model.predict(X_test)
predictions = std_sc.inverse_transform(predictions.reshape(-1,1)) # We have to undo the scaling we did to obtain the actual house prices that we have predicted.
predictions

"""### End of the code"""