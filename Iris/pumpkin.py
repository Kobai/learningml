import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

X = df.drop(df.columns[4], axis=1).values

Y = df[df.columns[4]].values
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(3, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(
    X,Y,
    epochs = 20,
    verbose = 0
)

score = model.evaluate(testX, testY, batch_size=128, verbose=1)
print("Test score: ", score[0])
print("Accuracy: ", score[1])