"""
In this section, we will train an agent that will perform reinforcement learning 
based on the actor and critic networks. We will perform the following steps to achine it.
"""
#1. Create an agent class whose initial function takes in the batch size, state size, and an evaluation Boolean function, to check whether the training is ongoing.
#2. In the agent class, create the following methods:

#3. Import Actor_script and Critic_script
from Actor_script import Actor
from Critic_script import Critic

#4.
import numpy as np 
from numpy.random import choice
import random

from collections import namedtuple, deque

#5. Create a ReplayBuffer class that adds, samples, and evaluates a buffer:
class ReplayBuffer:
    # Fixed size of buffer to stay experience tuples

    def __init__(self,buffer_size,batch_size):
        # initialize a replay buffer object.

        # Parameters
        # Buffer_size: Maximum size of buffer. Batch_size: size of each batch
        self.memory = deque(maxlen=buffer_size) # Memory size of replay buffer
        self.batch_size = batch_size # Training batch size for neural nets
        self.experience = namedtuple("Experience",field_names=['state','action','reward','next_state','done'])

    # Add new experience to replay buffer memory
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)

    # Randomly sample a batch of experienced tuples from the memory
    # It makes sure that the states that we feed to the model are not temporally correlated.
    # It reduces overfitting
    def sample(self,batch_size=32):
        return random.sample(self.memory,k = self.batch_size)
    
    # Return current size of the buffer memory
    def __len__(self):
        return len(self.memory)
    
# The reinforcement learning agent that learns using the actor-critic network is:
class Agent:
    def __init__(self,state_size,batch_size,is_eval=False):
        self.state_size = state_size

        # the number of actions are defined as 3: sit, buy, sell
        self.action_size = 3

        # Define replay memory size
        self.buffer_size = 1000000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size,self.batch_size)
        self.inventory = []

        # Define whether or not training is on going. This variable will be changed during the training and evaluation phrase
        self.is_eval = is_eval

        # Discount factor in Bellman equation:
        self.gamma = 0.99
        # A soft update of the actor and critic networks can be done as following:
        self.tau = 0.001

        # The actor policy model maps states to actions and instantiates the actor networks
        # Local and target models, for soft updates of parameters
        self.actor_local = Actor(self.state_size,self.action_size)
        self.actor_target = Actor(self.state_size,self.action_size)

        # The critic(value) model that maps the state-action pairs to Q-values
        self.critic_local = Critic(self.state_size,self.action_size)

        self.critic_target = Critic(self.state_size,self.action_size)
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Set the target model parameters to local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

    # Run an action, given a state, using the actor(policy network) and the output of
    # the softmax layer of the actor-network , returning the probabiliry for each action
    def act(self,state):
        options = self.actor_local.model.predict(state)
        self.last_state = state
        if not self.is_eval:
            return choice(range(3),p=options[0])
        return np.argmax(options[0])
    
    # Return a stochastic policy, based on the action probability in the training model
    # and a deterministic action corresponding to the maximum probability during the test
    def step(self,action,reward,next_state,done):
        self.memory.add(self.last_state,action,reward,next_state,done)

        if len(self.memory)> self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            self.last_state = next_state
    
    # Learning from the sampled experience through the actor and the critic.
    # Create a method to learn from the sampled experience through the actor and critic
    def learn(self,experiences):
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)

        # Return a seperate array for each experience in the replay component and predict actions based on the next states
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        # Predict Q_values of the actor output for the next state
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states,actions_next])
        # Target the Q_values to serve as label for the critic network
        Q_targets = rewards + self.gamma * Q_targets_next * (1-dones)
        # Fit the critic model to the time difference
        self.critic_local.model.train_on_batch(x=[states,actions],y=Q_targets)

        # Train the actor model (local) using the gradient of the critic network output
        # with the respect to the action probabilities fed from the actor-network.
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states,actions,0]),(-1,self.action_size))

        # Define a custom training function
        self.actor_local.train_fn([states,action_gradients,1])
        # Initial a soft update of the parameters of both networks
        self.soft_update(self.actor_local.model,self.actor_target.model)

    def soft_update(self,local_model,target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights)
        new_weights = self.tau * local_weights + (1-self.tau)*target_weights
        target_model.set_weights(new_weights)

        

