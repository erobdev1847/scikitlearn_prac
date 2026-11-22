import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score


data = {
  'company_name': [
    'TechNova', 'Cloudify', 'AI Dynamics', 'ShopEase', 'HealthPlus',
    'DataWorks', 'FinTech Solutions', 'GreenAI', 'EcomMax', 'MedSys',
    'CodeCrafters', 'VisionCloud', 'NextGenHealth', 'AlgoRise', 'SmartTrade',
    'CyberCore', 'BrightSoft', 'UrbanTech', 'HealthLink', 'QuantumLeap'
  ],
  'founding_date': [
    '06/15/2012', '08/23/2017', '01/10/2010', '12/05/2018', '04/20/2015',
    '03/14/2011', '09/30/2014', '11/18/2019', '07/22/2013', '05/03/2016',
    '10/28/2009', '02/11/2018', '06/02/2012', '04/17/2015', '09/08/2017',
    '01/25/2010', '07/10/2013', '10/06/2016', '03/29/2014', '11/12/2011'
  ],
  'funding_rounds': [
    4, 2, 6, 3, 5,
    3, 4, 1, 3, 5,
    7, 2, 5, 4, 3,
    6, 3, 2, 4, 7
  ],
  'total_funding_million': [
    120.5, np.nan, 340.0, 60.0, 150.0,
    75.0, 200.0, 10.0, 55.0, np.nan,
    500.0, 30.0, 160.0, 95.0, 45.0,
    300.0, 80.0, 25.0, 110.0, 600.0
  ],
  'annual_revenue_million': [
    85.3, 20.1, np.nan, 45.0, 90.0,
    40.0, 130.0, 5.5, 35.0, 75.0,
    400.0, 18.0, 100.0, np.nan, 22.0,
    220.0, 60.0, 12.0, 95.0, 450.0
  ],
  'employee_count': [
    320, 80, 1100, np.nan, 400,
    200, 700, 50, 150, 350,
    2300, 90, 500, 400, 120,
    1800, np.nan, 70, 450, 2800
  ],
  'industry_sector': [
    'Fintech', 'Software', 'AI', 'Ecommerce', 'Healthtech',
    'AI', 'Fintech', 'AI', 'Ecommerce', 'Healthtech',
    'Software', 'Software', 'Healthtech', 'AI', 'Fintech',
    'Software', 'Software', 'Software', 'Healthtech', 'AI'
  ],
  'ceo_experience_years': [
    15, 8, 20, 5, 18,
    12, 16, 4, 7, 19,
    25, 9, 20, 17, 6,
    22, 11, 5, 18, 30
  ],
  'profitability': [
    1, 0, 1, 0, 1,
    1, 1, 0, 0, 1,
    1, 0, 1, 1, 0,
    1, 0, 0, 1, 1
  ],
  'ipo_next_year': [
    1, 0, 1, 0, 1,
    0, 1, 0, 0, 1,
    1, 0, 1, 1, 0,
    1, 0, 0, 1, 1
  ]
}

df = pd.DataFrame(data)
df['industry_sector']=df['industry_sector'].str.lower()
y = df['ipo_next_year']
X = df.drop(columns = ['company_name','founding_date','ipo_next_year'])
# initial x an y set. Corrected mistake here.
# Originally altered the dataframe to fill nans with column averages, which is a data leak;
#  wait to fill nans with averages from training sets only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)
X_train_num = X_train.drop(columns = ['industry_sector'])
X_test_num = X_test.drop(columns = ['industry_sector'])
train_means = X_train_num.mean()
X_train_num = X_train_num.fillna(train_means)
X_test_num = X_test_num.fillna(train_means)
ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X_train[['industry_sector']])
X_train_enc = ohe.transform(X_train[['industry_sector']])
X_test_enc = ohe.transform(X_test[['industry_sector']])
enc_columns = ohe.get_feature_names_out()
X_train_enc = pd.DataFrame(X_train_enc, columns = enc_columns)
X_test_enc = pd.DataFrame(X_test_enc, columns = enc_columns)
# Remember to convert back into dataframe. Since this is one column it saves as an array
# so you have to remember to manually convert back to a dataframe
scaler = MinMaxScaler()
scaler.fit(X_train_num)
scaled_array_train = scaler.transform(X_train_num)
scaled_array_test = scaler.transform(X_test_num)
X_scaled_train = pd.DataFrame(scaled_array_train, columns = X_train_num.columns)
X_scaled_test = pd.DataFrame(scaled_array_test, columns = X_test_num.columns)
X_train_comp = pd.concat([X_train_enc, X_scaled_train], axis = 1)
X_test_comp = pd.concat([X_test_enc, X_scaled_test], axis = 1)
model = KNeighborsClassifier()
model.fit(X_train_comp, y_train)
y_pred = model.predict(X_test_comp)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(y_pred)
ipo_companies = df.loc[X_test.index[y_pred==1], 'company_name']
print(ipo_companies.to_string())


