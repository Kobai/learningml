import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout

env = gym.make('CartPole-v1')
env.reset()
random.seed()

''' used to find input dim => 4
action = env.action_space.sample()
observation,reward,done,_ = env.step(action)
print(env.observation_space.shape)
'''

IN = 4
OUT = 2 # left or right

model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, input_shape=(4,), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, input_shape=(4,), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, input_shape=(4,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, input_shape=(4,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ['accuracy']
)

