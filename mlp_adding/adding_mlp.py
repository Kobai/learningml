import pandas as pd
from keras.models import Sequential
from keras.layers import *
import math

# Neural Network that learns how to add two numbers. It's still a little inaccurate, but it's better than me lol
# The loss by epoch 60 is usually in the order of 10^(-8) or better.

# Grabbing training data
training_data_df = pd.read_csv("training.csv")
X = training_data_df.drop('numsum', axis=1).values /2000
Y = training_data_df[['numsum']].values /2000

# Build network with a "little bit" of guessing and checking
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Trained the neural net
model.compile(loss='mse', optimizer = 'adam')

model.fit(
    X,Y,
    epochs = 60,
    shuffle=True,
    verbose = 2
)

# Time to test the neural net
test_data_df = pd.read_csv("testing.csv")
X_test = test_data_df.drop('numsum', axis=1).values /2000
Y_test = test_data_df[['numsum']].values /2000

test_error_rate = model.evaluate(X_test, Y_test, verbose=1)
print("Test Error Rate: ", test_error_rate)

model.save("trained_model.h5")
print("Model saved")






