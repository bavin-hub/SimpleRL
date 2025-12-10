import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import gym
import random
from torch.distributions.normal import Normal
import itertools
import torch.nn.functional as F




class Memory:
    def __init__(self, batch_size, memory_size, env, device):
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.mem_ctr = 0
        self.state_shape = np.array(env.observation_space.shape).prod()
        self.action_shape = np.array(env.action_space.shape).prod()
        self.initialize_memory()
    
    def initialize_memory(self):
        self.states = t.zeros(self.memory_size, self.state_shape, device=self.device)
        self.actions = t.zeros(self.memory_size, self.action_shape, device=self.device)
        self.rewards = t.zeros(self.memory_size, device=self.device)
        self.next_states = t.zeros(self.memory_size, self.state_shape, device=self.device)
        self.dones = t.zeros(self.memory_size, dtype=t.bool ,device=self.device)


    def generate_batches(self):
        max_mem = min(self.mem_ctr, self.memory_size)
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, next_states, dones


    def store_memory(self, state, action, reward, next_state, done):
        idx = self.mem_ctr % self.memory_size
        # if idx < self.memory_size:
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.mem_ctr += 1



class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.input_dims= input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(self.input_dims+self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer= optim.Adam(self.parameters(), lr=self.beta)

        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(self.device)


    def forward(self, state, action):
        q1_action_value = self.fc1(t.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1 = self.q1(q1_action_value)

        return q1
    


class ActorNetwork(nn.Module):
    def __init__(self, alpha, fc1_dims, fc2_dims, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(self.device)

    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        prob = t.tanh(self.mu(prob))

        return prob
    



class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions, tau, env, gamma=0.99, max_size=100000, 
                 fc1_dims=400, fc2_dims=300, batch_size=100, d=2, warmup=1000, noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = Memory(batch_size, max_size, env, 'cuda')
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.learn_step_ctr = 0
        self.time_step = 0
        self.update_actor_iter = d

        self.warmup_steps = warmup

        self.actor = ActorNetwork(alpha, fc1_dims, fc2_dims, input_dims, n_actions)
        self.critic1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
        self.critic2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)

        self.target_actor = ActorNetwork(alpha, fc1_dims, fc2_dims, input_dims, n_actions)
        self.target_critic1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
        self.target_critic2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)

        self.update_network_parameters(1)

        # print()
        
        self.noise = noise

        self.device = 'cuda' if t.cuda.is_available() else 'cpu'


    def update_network_parameters(self, tau=None):
        if tau==None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic1.named_parameters()
        target_critic2_params = self.target_critic2.named_parameters()

        
        actor_params_dict = dict(actor_params)
        critic1_params_dict = dict(critic1_params)
        critic2_params_dict = dict(critic2_params)

        target_actor_params_dict = dict(target_actor_params)
        target_critic1_params_dict = dict(target_critic1_params)
        target_critic2_params_dict = dict(target_critic2_params)
            

        for name in actor_params_dict:
            actor_params_dict[name] = tau * actor_params_dict[name].clone() + \
                                    (1-tau)*target_actor_params_dict[name].clone()
            
        for name in critic1_params_dict:
            critic1_params_dict[name] = tau * critic1_params_dict[name].clone() + \
                                    (1-tau)*target_critic1_params_dict[name].clone()
            
        for name in critic2_params_dict:
            critic2_params_dict[name] = tau * critic2_params_dict[name].clone() + \
                                    (1-tau)*target_critic2_params_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_params_dict)
        self.target_critic1.load_state_dict(critic1_params_dict)
        self.target_critic2.load_state_dict(critic2_params_dict)

        

    def choose_action(self, state):
        if self.time_step < self.warmup_steps:
            mu = t.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
            mu = t.tensor(np.random.uniform(self.min_action, self.max_action)).to(self.device)
        else:
            state = t.tensor([state]).to(self.device)
            mu = self.actor.forward(state).view(-1)
        
        mu_prime = mu + t.tensor(np.random.normal(scale=0.2, size=mu.shape), dtype=t.float).to(self.device)
       
        mu_prime = t.clamp(mu_prime, t.tensor(self.min_action).to(self.device), 
                                     t.tensor(self.max_action).to(self.device))
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)


    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.generate_batches()

        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + t.clamp(t.tensor(np.random.normal(scale=0.2, size=target_actions.shape), dtype=t.float).to(self.device), -0.5, 0.5)
        target_actions = t.clamp(target_actions, t.tensor(self.min_action).to(self.device),
                                                 t.tensor(self.max_action).to(self.device))

        q1_ = self.target_critic1.forward(next_state, target_actions)
        q2_ = self.target_critic2.forward(next_state, target_actions)

        q1 = self.critic1.forward(state, action)
        q2 = self.critic2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_= t.min(q1_, q2_)
        # print(critic_value_.shape)
        target = reward + self.gamma * critic_value_
        # print(self.batch_size)
        target = target.view(self.batch_size, 1)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_ctr += 1

        if self.learn_step_ctr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_loss = self.critic1.forward(state, self.actor.forward(state))
        actor_loss = -t.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()



def run():
    env_id = 'BipedalWalker-v3'
    env = gym.make(env_id)

    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = np.array(env.action_space.shape).prod()

    agent = Agent(alpha=0.001, beta=0.001, input_dims=input_dims, n_actions=n_actions,
                  tau=0.005, env=env)
    
    best_score = env.reward_range[0]
    score_history = []

    steps = 0
    n_games = 100000
    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(t.tensor(state), t.tensor(action), reward, t.tensor(next_state), done)
            agent.learn()
            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        print(f'episode : {i}, score : {score}, avg_score : {avg_score}')
        



if __name__=='__main__':
    run()  

        