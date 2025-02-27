#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("zomato_cleaned.csv")
df.head(3)


# In[3]:


df.describe()


# In[4]:


df.drop(columns=['name', 'city'], inplace=True)


# ## One hot encoding

# In[6]:


# Encode categorical features
cat_cols = ['location', 'rest_type', 'type']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# In[7]:


# double check if any features are not in numeric
print(df.dtypes[df.dtypes == 'object'])


# ## MultiLabelBinarizer

# In[9]:


from sklearn.preprocessing import MultiLabelBinarizer

# split multiple categories
df['cuisines'] = df['cuisines'].str.split(', ')

mlb = MultiLabelBinarizer()
cuisine_encoded = pd.DataFrame(mlb.fit_transform(df['cuisines']), columns=mlb.classes_)

# Merge back with original dataframe
df = df.join(cuisine_encoded).drop(columns=['cuisines'])


# ## Top 20 dishes encoding

# In[11]:


from collections import Counter

# Count most common dishes
dish_counter = Counter(', '.join(df['dish_liked'].dropna()).split(', '))
top_20_dishes = [dish for dish, count in dish_counter.most_common(20)]

# Create binary columns for top dishes
for dish in top_20_dishes:
    df[dish] = df['dish_liked'].apply(lambda x: 1 if pd.notna(x) and dish in x else 0)

# Drop the original 'dish_liked' column
df.drop(columns=['dish_liked'], inplace=True)


# In[12]:


df.head(3)


# ## Data splitting

# In[14]:


from sklearn.model_selection import train_test_split

X = df.drop(columns=['rate'])
y = df['rate']

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Model training
# ## Linear regression

# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression - MSE:", mse_lr)
print("Linear Regression - R² Score:", r2_lr)


# ## Random forest regressor

# In[18]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest - MSE:", mse_rf)
print("Random Forest - R² Score:", r2_rf)


# ## XG Boost regressor
!pip install xgboost
# In[20]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost - MSE:", mse_xgb)
print("XGBoost - R² Score:", r2_xgb)


# ## Model Evaluation

# In[22]:


import matplotlib.pyplot as plt

models = ['Linear Regression', 'Random Forest', 'XGBoost']
mse_scores = [mse_lr, mse_rf, mse_xgb]
r2_scores = [r2_lr, r2_rf, r2_xgb]

plt.figure(figsize=(12, 5))

# MSE
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")

# R²
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.show()


# ## Feature Selection & Importance using best performed model (random forest)

# In[24]:


feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)

# top 10 important features
plt.figure(figsize=(10, 6))
feature_importance.nlargest(10).plot(kind='barh', color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features in Prediction")
plt.show()


# # Fine Tune Models
# ## Random Forest with GridSearchCV

# In[26]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Tuned Random Forest - MSE:", mse_rf)
print("Tuned Random Forest - R² Score:", r2_rf)


# ## XGBoost with RandomizedSearchCV

# In[28]:


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, cv=3, scoring='r2', n_jobs=-1, n_iter=10)
xgb_search.fit(X_train, y_train)

print("Best XGBoost Parameters:", xgb_search.best_params_)

best_xgb_model = xgb_search.best_estimator_

y_pred_xgb = best_xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("Tuned XGBoost - MSE:", mse_xgb)
print("Tuned XGBoost - R² Score:", r2_xgb)


# ## Model Evaluation Fine Tuned

# In[30]:


import matplotlib.pyplot as plt

models = ['Tuned Random Forest', 'Tuned XGBoost']
mse_scores = [mse_rf, mse_xgb]
r2_scores = [r2_rf, r2_xgb]

plt.figure(figsize=(12, 5))

# MSE
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['green', 'red'])
plt.xlabel("Models")
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")

# R² Score
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['green', 'red'])
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.show()

