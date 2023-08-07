import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns

import math
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/gbpusd.csv', sep='\t', encoding='utf-8')
df = pd.DataFrame.from_dict({"timestamp": df['timestamp'], 'ask': df['ask'], 'bid': df['bid']})

# Make different frequency of tick data
df['group_id_50'] = df.index // 50
df['group_id_100'] = df.index // 100
df['group_id_500'] = df.index // 500
df['group_id_1000'] = df.index // 1000
df['mid_price'] = (df['bid'] + df['ask']) / 2

df.head()


grouped_50 = df.groupby('group_id_50')
price_50_ohlc = grouped_50['mid_price'].ohlc()
price_50_ohlc['timestamp'] = grouped_50['timestamp'].first()

grouped_100 = df.groupby('group_id_100')
price_100_ohlc = grouped_100['mid_price'].ohlc()
price_100_ohlc['timestamp'] = grouped_100['timestamp'].first()

grouped_500 = df.groupby('group_id_500')
price_500_ohlc = grouped_500['mid_price'].ohlc()
# price_500_ohlc['timestamp'] = grouped_500['timestamp'].first()

grouped_1000 = df.groupby('group_id_1000')
price_1000_ohlc = grouped_1000['mid_price'].ohlc()
price_1000_ohlc['timestamp'] = grouped_1000['timestamp'].first()


dataset = np.array(price_1000_ohlc['close']).reshape(-1, 1)

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(dataset)
dataset = scaler.transform(dataset)

train_size = int(len(dataset) * .8)
test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Given input shape. This needs to be adjusted according to your specific dataset.
input_shape = (5, 1)  # (timesteps, features)


#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 5  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(train) - n_future +1):
    trainX.append(train[i - n_past:i, 0])
    trainY.append(train[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


#Empty lists to be populated using formatted training data
testX = []
testY = []

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(test) - n_future +1):
    testX.append(test[i - n_past:i, 0])
    testY.append(test[i + n_future - 1:i + n_future, 0])

testX, testY = np.array(testX), np.array(testY)

print('trainX shape == {}.'.format(testX.shape))
print('trainY shape == {}.'.format(testY.shape))


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential([
    Conv1D(filters=32, kernel_size=1, padding='same', activation='relu', input_shape=(None, n_past)),
    MaxPooling1D(pool_size=3, padding='same'),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    # Replace N with the number of classes you have in case of classification, or 1 for regression.
    Dense(7, activation='softmax')  # 'softmax' for classification, 'linear' for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.2, verbose=1)


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))

print('Train Score: %.2f RMSE.' % trainScore)

testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))

print('Test Score: %.2f RMSE.' % testScore)






