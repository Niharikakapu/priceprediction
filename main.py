#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures


# In[4]:


import pandas as pd
df = pd.read_csv(r'C:\Users\leela\Downloads\kc_house_data.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe().T


# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\leela\Downloads\data.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Example 1: Bar Chart
plt.figure(figsize=(10, 6))
df['Column_Name'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Bar Chart - Distribution of Column_Name')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.show()

# Example 2: Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Column1'], df['Column2'], color='orange')
plt.title('Scatter Plot - Column1 vs Column2')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.show()


# In[10]:


print("Missing Values by Column")
print("-"*30)
print(df.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",df.isna().sum().sum())


# In[ ]:


xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(xgb)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "XGBRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = models.append(new_row, ignore_index=True)
