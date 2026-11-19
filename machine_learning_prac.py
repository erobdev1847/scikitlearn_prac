import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import csv
df = pd.read_csv('AI_Usage_and_Impact_on_Students_and_Professionals.csv')
df['AI_Usage_Hours_Per_Day']  = (
    df['AI_Usage_Hours_Per_Day']
    .astype(str)
    .str.replace('hrs', '', case=False, regex=False)
    .str.strip()
    .astype(float)
)
df = df.dropna(subset = ['AI_Usage_Hours_Per_Day', 'Productivity_Score'])
df = df[df['AI_Usage_Hours_Per_Day']<15].copy()
print(df['AI_Usage_Hours_Per_Day'].nlargest(10))
X = df[['AI_Usage_Hours_Per_Day']]
y = df['Productivity_Score']
poly = PolynomialFeatures(degree = 2, include_bias = False)
model = LinearRegression()
X_sorted = X.sort_values('AI_Usage_Hours_Per_Day')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, random_state = 42)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_poly = poly.transform(X_sorted)
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)



residuals = y_test - y_pred
fig, axes = plt.subplots(2,1, figsize=(8,8))
axes[0].scatter(y_pred, residuals)
axes[1].plot(X_sorted, model.predict(X_poly), color = 'red')
axes[0].axhline(y=0)
axes[0].set_xlabel('Predictions')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predictions')
axes[1].scatter(X,y)
axes[1].set_xlabel('Independent Variable')
axes[1].set_ylabel('Dependent Variable')
axes[1].set_title('Actual Data')
plt.show()



