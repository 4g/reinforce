"""
Use a network to learn (environment1, action) -> (environment2, reward)
Then use the network to get the correct action. 
There are only 2 possible actions.
"""

import gym
from keras.layers import Dense, Input, concatenate
from keras.models import Model
import numpy as np

env = gym.make('CartPole-v0')

def get_random_episode():
  observations, actions, rewards = [], [], []
  observation = env.reset()

  for t in range(100):
    observations.append(observation)
    #env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    actions.append(action)
    rewards.append(reward)
    if done:
      rewards[-1] = 0.0
      return (observations, actions, rewards)

# Create random episodes
num_episodes = 100000
episodes = []
O = []
A = []
Y = []
for i in range(num_episodes):
  x = get_random_episode()
  # print x
  if x is None:
    continue
  o,a,r = x
  for j in range(len(o)):
    O.append(o[j])
    A.append(a[j])
    Y.append(r[j])

observation_size = len(O[0])
action_size = 1

O = np.asarray(O)
A = np.asarray(A)
Y = np.asarray(Y)

# Create the model graph
input1 = Input(shape=(observation_size,))
input2 = Input(shape=(action_size,))
concat_input = concatenate(inputs=[input1, input2])
dense1 = Dense(32)(concat_input)
dense2 = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[input1, input2], outputs=[dense2])

print model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit([O, A], Y, epochs=10, batch_size=1024,validation_split=0.5,shuffle=True,class_weight={0:.95, 1:.05})

error = .95
