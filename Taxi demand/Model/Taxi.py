import sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

data = pd.read_csv("./dataset/FinalData_for_Models.csv")
data.rename(columns={'Unnamed: 0':'pickup_time'}, inplace=True)
print(data.head())
data = data.loc[data.missing_dt == False, :]
data.drop("missing_dt", axis=1, inplace=True)
print(data.shape)
data_wm_dummies = data['weather_main'].str.split(",").str.join("*").str.get_dummies(sep='*')
data_wd_dummies = data['weather_description'].str.split(",").str.join("*").str.get_dummies(sep='*')
data.drop(["weather_main", "weather_description"], axis=1, inplace=True)
data = pd.concat([data, data_wm_dummies], axis=1)
print(data.shape)
print(data.head())
data['holiday'] = data.holiday.astype(int)
print(data.head())
data.rename(columns={'Hour':'HourOfDay'}, inplace=True)
data.rename(columns={'Day':'DayOfWeek'}, inplace=True)
print(data.head(2))
print(data.shape)
data.drop(["num_passengers"], axis=1, inplace=True)
data.drop(['Cancelled_Arriving_Flights'], axis=1, inplace=True)
data.drop(['Avg_Delay_Departing'], axis=1, inplace=True)
data.drop(['temp_min', 'temp_max'], axis=1, inplace=True)
print(data.head())
data.drop(['Month', 'HourOfDay', 'DayOfWeek'], axis=1, inplace=True)
print(data.head())
data.set_index("pickup_time", inplace=True)
print(data.head())
num_pickups = data.num_pickups
data.drop("num_pickups", axis=1, inplace=True)
np.random.seed(7)

def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

features_and_target = pd.concat([data, num_pickups], axis=1)
values = features_and_target.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
n_hours = 23
reframed = series_to_supervised(scaled, n_hours, 1)
n_features = features_and_target.shape[1]
print(reframed.head())
print(reframed.shape)
values = reframed.values
n_train_hours = 365 * 24 * 3
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print(test.shape)
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(24, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
print("test data")
print(test_X)
history = model.fit(train_X, train_y, epochs=10, batch_size=100,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
history_loss = pd.DataFrame()
history_loss['train'] = history.history['loss']
history_loss.plot(figsize=(10,10), fontsize=16,
                title='LSTM Model Loss');
plt.savefig("./images/lstm_model_loss.png")
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], -1))
print(test.shape)
print(test_X.shape) 
inv_yhat = np.concatenate((test[:, 345:359], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test[:, 345:359], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
inv_yhat_gte_zero = inv_yhat.copy()
inv_yhat_gte_zero[inv_yhat_gte_zero < 0] = 0
print("R2:  ", r2_score(inv_y, inv_yhat))
print("MAE: ", mean_absolute_error(inv_y, inv_yhat))
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print("RMSE:", rmse)
results = pd.DataFrame()
results['prediction'] = inv_yhat
results['actual'] = inv_y
joblib.dump(model, "./model/model_rfgbtxgb_la.pkl")

import seaborn as sns
sns.lmplot(x='actual',y='prediction',data=results,fit_reg=False)
plt.plot(results['actual'], results['actual'], color="orange", linewidth=2)
plt.title("Predicted Number of Taxi Pickups vs. Actual Number of Taxi Pickups")
plt.tight_layout()
plt.savefig("./images/lstm_model_pred_actual.png")
results.plot(figsize=(200,10));
print(results.head())
print(results.shape)
features_and_target.index[-4363:]
results.index = pd.Series(features_and_target.index[-4344:]).apply(lambda x: x.split(":00-0")[0])
pickups_gte_800 = results[results['actual'] > 800].copy()
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pickups_gte_800['day_of_week'] = list(pd.Series(pickups_gte_800.index).apply(lambda x: days_of_week[pd.to_datetime(x).dayofweek]))
pickups_gte_800['day_of_week'].value_counts().plot(kind='bar')
print(pickups_gte_800)
descriptions = [
    "President's Day",
    "Day after President's Day",
    "",
    "",
    "",
    "",
    "Easter",
    "Easter Monday",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
]
pickups_gte_800['description'] = descriptions
plt.show()
