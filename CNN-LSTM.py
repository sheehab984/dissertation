import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import seaborn as sns

import math
from sklearn.metrics import mean_squared_error

from helper import get_df_ohlc, split_data, transform_data_for_NN

# Load and prepare data.
df = pd.read_csv('data/gbpusd.csv', sep='\t', encoding='utf-8')
df = pd.DataFrame.from_dict({"timestamp": df['timestamp'], 'ask': df['ask'], 'bid': df['bid']})
df['mid_price'] = (df['bid'] + df['ask']) / 2

price_ohlc = get_df_ohlc(df, 1000)
dataset = np.array(price_ohlc['close']).reshape(-1, 1)

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(dataset)
dataset = scaler.transform(dataset)

train, test = split_data(dataset, 0.8)

# Given input shape. This needs to be adjusted according to your specific dataset.
input_shape = (5, 1)  # (timesteps, features)

seq_size = 5

trainX, trainY = transform_data_for_NN(train, seq_size)

testX, testY = transform_data_for_NN(test, seq_size)


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential([
    Conv1D(filters=32, kernel_size=1, padding='same', activation='relu', input_shape=(None, seq_size)),
    MaxPooling1D(pool_size=3, padding='same'),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    # Replace N with the number of classes you have in case of classification, or 1 for regression.
    Dense(1, activation='linear')  # 'softmax' for classification, 'linear' for regression
])

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='val_loss', # value being monitored for improvement
                          min_delta=0, # Abs value and is the min change required before we stop
                          patience=3, # Number of epochs we wait before stopping 
                          verbose=1,
                          mode='auto') # Direction that your monitor value should go

callbacks = [earlystop]


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy']) 

# fit the model
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.2, verbose=1, callbacks=callbacks)


trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(1, -1))

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(1, -1))

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

print('Train Score: %.6f RMSE.' % trainScore)

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

print('Test Score: %.6f RMSE.' % testScore)

# Convert these arrays into pandas dataframe
df = pd.DataFrame({'trainPredict': testPredict[:, 0].flatten(), 'trainY': testY.flatten()})

# Using seaborn to create a lineplot
plt.figure(figsize=(20,10))
sns.lineplot(data=df)
plt.title('Comparision Test of Predictions and Actual Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()


'''
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))

print('Train Score: %.2f RMSE.' % trainScore)

testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))

print('Test Score: %.2f RMSE.' % testScore)


length =  5

# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[length:len(trainPredict)+length, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
testPredictPlot[len(train)+(length)-1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

'''


