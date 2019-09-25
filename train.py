# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
import pandas as pd
import numpy as np

# IMPORT AZURE LIBRARY
from azureml.core.run import Run

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')

df["Date"] = pd.to_datetime(df["Date"])
df['Temp'] = df['Temp'].rolling(window=20).mean()

# Variable to predic
X = df.set_index('Date')

# fit model
model = SARIMAX(X,
                order=(0, 0, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# predict model
pred = model_fit.get_prediction(start='1989-12-31',end='1990-12-31', dynamic=False)

# evaluate model
y_forecasted = pred.predicted_mean
y_truth = X.loc[X.index >= '1989-12-31']
y_truth = pd.Series(y_truth['Temp'], index=X.index)

mse = ((y_forecasted - y_truth) ** 2).mean()
mse = round(mse, 2)
rmse = round(np.sqrt(mse), 2)
print('The Mean Squared Error is {}', mse)
print('The Root Mean Squared Error is {}', rmse)

# AZURE LOGGING VARIABLES
run_logger = Run.get_context()
run_logger.log(name='RMSE', value=rmse)
run_logger.log(name='MSE', value=mse)