#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

data=pd.read_csv('C:/Users/james/OneDrive/Documents/QQQ.csv')
data.head()

data['3d_future_close'] = data['Close'].shift(-3)
data.dropna(inplace=True)

X = data[['Volume', 'Low', 'Open', 'High', 'Oil_Price', 'Interest_Rate']]
y = data['3d_future_close']
feature_names = ['Volume', 'Low', 'Open', 'High', 'Oil_Price', 'Interest_Rate']

train_size = 800
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

grid = {'n_estimators': [200], 'max_depth': [13], 'max_features': [2, 10], 'random_state': [11]}
test_scores = []

rf_model = RandomForestRegressor()

for g in ParameterGrid(grid):
    rf_model.set_params(**g) 
    rf_model.fit(X_train, y_train)
    test_scores.append(rf_model.score(X_test, y_test))

best_index = np.argmax(test_scores)
print(test_scores[best_index], ParameterGrid(grid)[best_index])

rf_model = RandomForestRegressor(n_estimators=100, max_depth=13, max_features=11, random_state=11)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

y_pred_series = pd.Series(y_pred, index=y_test.index)
y_pred_series.plot()

plt.ylabel("Predicted Daily Close Price")
plt.xlabel("Dataset Observation Number")
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



importances = rf_model.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x_values = range(len(importances))
labels = np.array(feature_names)[sorted_index]
plt.bar(x_values, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




