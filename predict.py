import pandas as pd
# read dataset
sphist = pd.read_csv("sphist.csv")

# convert the Date column to a Pandas date type
from datetime import datetime
sphist['Date'] = pd.to_datetime(sphist['Date'], format='%Y-%m-%d')

# sort dataframe on Date column in ascending order
sphist.sort_values(by=['Date'], inplace=True)

print("This is our dataset:\n", sphist.head())

## Generating indicators

# creating past 5 days average index
sphist['avg_p5_days'] = (sphist['Close'].shift(1).rolling(window=5).sum()) / 5

# creating past 30 days average index
sphist['avg_p30_days'] = (sphist['Close'].shift(1).rolling(window=30).sum()) / 30

# creating past 365 days average index
sphist['avg_p365_days'] = (sphist['Close'].shift(1).rolling(window=365).sum()) / 365

# creating weighted average index between p5 and p365
sphist['wavg_p5_p365'] = 0.7*sphist['avg_p5_days'] + 0.3*sphist['avg_p365_days']

# creating past 5 days standard deviation index
sphist['std_p5_days'] = (sphist['Close'].shift(1).rolling(window=5).std())

# creating past 365 days standard deviation index
sphist['std_p365_days'] = (sphist['Close'].shift(1).rolling(window=365).std())

# creating weighted average of std devation between past 5 and 365 days
sphist['wavg_std_p5_p365'] = 0.2 * sphist['std_p5_days'] + 0.8 * sphist['std_p365_days']

## Splitting upt the data

# Remove any rows from the DataFrame that fall before 1951-01-03
sphist = sphist[sphist["Date"] > datetime(year=1951, month=1, day=2)]

# remove any rows with NaN values
sphist_new = sphist.dropna()

# Generate two new dataframes to use in making our algorithm, 2013-01-01 as the cutoff
train = sphist_new[sphist_new["Date"] < datetime(year=2013, month=1, day=1)]
test = sphist_new[sphist_new["Date"] >= datetime(year=2013, month=1, day=1)]

## Making predictions

# use linear regression class from sklearn to train a model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# selecting feature
# as we are dealing with time series data, we cannot use any of the original columns as features
features = ['avg_p5_days', 'wavg_p5_p365', 'std_p365_days']

# instantiate the model class
lr = LinearRegression()

# train model with feature on target
lr.fit(train[features], train["Close"])
# test model on test dataset
predictions = lr.predict(test[features])

# calculate MAE of model
lr_mae = mean_absolute_error(test["Close"], predictions)

print('Linear regression model with features:', features, '\nMean absolute error:', lr_mae)

# re-selecting features to improve model
new_feat = ['avg_p5_days', 'avg_p30_days', 'wavg_p5_p365']

# train model with new features on target
lr.fit(train[new_feat], train["Close"])
# test model on test dataset
predictions = lr.predict(test[new_feat])

# calculate MAE of model
lr_mae = mean_absolute_error(test["Close"], predictions)

print('Linear regression model with new features:', new_feat, '\nMean absolute error:', lr_mae)

# selecting features
new_feat = ['avg_p5_days', 'avg_p30_days', 'wavg_p5_p365']

# train model with new features on target
lr.fit(train[new_feat], train["Close"])
# test model on test dataset
predictions = lr.predict(test[new_feat])

# calculate MAE of model
lr_mae = mean_absolute_error(test["Close"], predictions, sample_weight=test['wavg_std_p5_p365'])

print('Linear regression model with adjusted sample weights, using features:', new_feat, '\nMean absolute error:', lr_mae)