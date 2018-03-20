# -*- coding: iso-8859-15 -*-

"""
--Learning to balance a pole--

learn a q-network, which learning the q function directly from (s,a,r,s')
http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
As per https://arxiv.org/pdf/1312.5602.pdf
"""

import gym
from keras.layers import  Dense, Input, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

random.seed(9001)

env = gym.make('CartPole-v0')
print dir(env)

class Agent:
  def __init__(self, action_space, sample_observation):
    self.memory_size = 10000
    self.events = deque(maxlen=self.memory_size)
    self.action_space = action_space
    self.model = self.q_network(action_space.size, sample_observation.size)
    # agent's age increases whenever
    # an episode is over
    # it also decreases the exploration probability when that happens
    # since episode's length increase with time, agent ages slowly 
    self.age = 0
    self.exploration_prob = 1.0
    self.learning_sample_size = 40
    
  def q_network(self, action_size, observation_size):
    # lets follow the structure of network used 
    # in deepminds atari network, where state is mapped 
    # to q-values
    
    mdl = Sequential()
    mdl.add(Dense(32, input_shape=(observation_size,), activation='relu'))
    mdl.add(Dense(32, activation='relu'))
    mdl.add(Dense(action_size, activation='linear'))
    mdl.compile(loss='mse', optimizer=Adam(0.001))
    print mdl.summary()
    return mdl
  
  def action(self, state):
    # explore vs exploit
    # we want to agent to do 
    # exploration in the beginning
    # and later go towards exploitation
    # or we can control the rate on basis 
    # of error
    if np.random.rand() > self.exploration_prob:
      action_probs = self.model.predict(state)
      action = self.action_space[np.argmax(action_probs)]
      
    else:
      action = random.randrange(self.action_space.size)
    
    return action
  
  def learn(self):
    # As per https://arxiv.org/pdf/1312.5602.pdf
    # learning directly from consecutive samples is inefficient, due to the strong correlations
    # between the samples; randomizing the samples breaks these correlations and therefore reduces the
    # variance of the updates.
    gamma = 0.95
    if len(self.events) < self.learning_sample_size:
      return
    training_samples = random.sample(self.events, self.learning_sample_size)
    
    # From Bellman's equation
    # Q(s,a) = r + γ * (max of Q(s′,a′) for all a')
    # Loss = Predicted Q - Real Q 
    # predicted Q = max Q at a state amongst all actions
    # Q(s,a) = r + γ * model.predict(s')
    # Loss = model.predict(s) - (r + γ * model.predict(s'))
    
    for sample in training_samples:
    
      state, action, reward, observation, done = sample
      current_values = self.model.predict(state)
      
      if not done:
        expected_value = reward + gamma * max(self.model.predict(observation)[0])
      else:
        expected_value = -5.0
      # we only propogate loss from the chosen (state,action)
      
      current_values[0][action] = expected_value
      # print state, current_values
      
      self.model.fit(state, current_values, verbose=0)
      
    self.age += 1
    self.exploration_prob = self.exploration_prob * .995
    return
    
  def save(self, state, action, reward, observation, done):
    self.events.append([state, action, reward, observation, done])

def play(max_games):
  sample_state = env.reset()
  action_space = np.asarray(range(env.action_space.n))
  agent = Agent(action_space, sample_state)
  past_games = deque(maxlen=100)
  num_games = 0
  
  while num_games < max_games:
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    game_length = 0
    while True:
      game_length += 1
      action = agent.action(state)
      observation, reward, done, info = env.step(action)
      if num_games > 1000:
        env.render()
      
      # save s,a,r,s' in memory of the agent
      # don't train the agent directly
      # later use the memory to replay and learn
      # i.e. learning happens at the end of episode 
      observation = np.expand_dims(observation, axis=0)
      agent.save(state, action, reward, observation, done)
      state = observation
      if done:
        break
    
    # episode is over
    # learn by replay  
    agent.learn()
    num_games += 1
    past_games.append(game_length)
    print num_games, game_length,  agent.exploration_prob, sum(past_games)/100.0
      
if __name__ == '__main__':
  play(5000)
