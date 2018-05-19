import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)
df = df.drop("PassengerId", axis=1)

df["Sex"] = df["Sex"].map({'male':1, 'female':0})
df["Embarked"] = df["Embarked"].map({'C':1,'Q':2,'S':3})

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mean())

print(df.columns[df.isnull().any()])

X = df.drop("Survived",axis=1).values / 520
Y = df[['Survived']].values

(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(
    trainX,trainY,
    epochs=100,
    verbose = 2
)

score = model.evaluate(testX, testY, batch_size=128, verbose=1)
print("Test score:", score[0])
print("Accuracy: ", score[1])

df = pd.read_csv("test.csv")

df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)

df["Sex"] = df["Sex"].map({'male':1, 'female':0})
df["Embarked"] = df["Embarked"].map({'C':1,'Q':2,'S':3})

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

X = df.drop("PassengerId",axis=1).values / 520

y = model.predict(X).round()
y = pd.DataFrame(y)
y.columns = ['Survived']

out = pd.concat([df[['PassengerId']],y], axis=1)

out.to_csv("out.csv", index=False)

