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

model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ['accuracy']
)

# Parameters
gamma = 0.99
epsilon = 0.05
episodes = 400
maxsteps = 500
scores = []
memory = []
        
for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1,4])
    for j in range(maxsteps):
        action = 0
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action =  np.argmax(model.predict(state)[0])
        next_state, reward,done,_ = env.step(action)
        next_state = np.reshape(next_state, [1,4])        
        memory.append((state,action, reward, next_state,done))
        state = next_state
        if done:
            print("episodeL {}/{}, score: {}".format(i, episodes, j))
            break
    minibatch = random.sample(memory, min(32,len(memory)))
    for state,action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma + np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > 0.01:
        epsilon *= 0.995        
            
        