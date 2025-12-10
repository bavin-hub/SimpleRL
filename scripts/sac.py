# import os
# import numpy as np
# import torch as t
# import torch.nn as nn
# import torch.optim as optim
# import gym
# import random
# from torch.distributions.normal import Normal
# import itertools
# import torch.nn.functional as F



# class Memory:
#     def __init__(self, batch_size, memory_size, env, device):
#         self.device = device
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.mem_ctr = 0
#         self.state_shape = np.array(env.observation_space.shape).prod()
#         self.action_shape = np.array(env.action_space.shape).prod()
#         self.initialize_memory()
    
#     def initialize_memory(self):
#         self.states = t.zeros(self.memory_size, self.state_shape, device=self.device)
#         self.actions = t.zeros(self.memory_size, self.action_shape, device=self.device)
#         self.rewards = t.zeros(self.memory_size, device=self.device)
#         self.next_states = t.zeros(self.memory_size, self.state_shape, device=self.device)
#         self.dones = t.zeros(self.memory_size, dtype=t.bool ,device=self.device)


#     def generate_batches(self):
#         max_mem = min(self.mem_ctr, self.memory_size)
#         batch = np.random.choice(max_mem, self.batch_size)

#         states = self.states[batch]
#         actions = self.actions[batch]
#         rewards = self.rewards[batch]
#         next_states = self.next_states[batch]
#         dones = self.dones[batch]

#         return states, actions, rewards, next_states, dones


#     def store_memory(self, state, action, reward, next_state, done):
#         idx = self.mem_ctr % self.memory_size
#         # if idx < self.memory_size:
#         self.states[idx] = state
#         self.actions[idx] = action
#         self.rewards[idx] = reward
#         self.next_states[idx] = next_state
#         self.dones[idx] = done
#         self.mem_ctr += 1



# class CriticNetwork(nn.Module):
#     def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
#                  ):
#         super(CriticNetwork, self).__init__()

#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
        
#         self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.q1 = nn.Linear(self.fc2_dims, 1)

#         self.optimizer = optim.Adam(self.parameters(), lr=beta)
#         self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        
#         self.to(self.device)
    
#     def forward(self, state, action):
#         action_value = self.fc1(t.cat([state, action], dim=1))
#         action_value = F.relu(action_value)
#         action_value = self.fc2(action_value)
#         action_value = F.relu(action_value)
#         q1 = self.q1(action_value)
#         return q1
    

# class ActorNetwork(nn.Module):
#     def __init__(self, alpha, input_dims, n_actions, fc1_dims, fc2_dims, max_action):
#         super(ActorNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.n_actions = n_actions
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.reparam_noise = 1e-6
#         self.max_action = max_action
        
#         self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.mu = nn.Linear(self.fc2_dims, self.n_actions)
#         self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = 'cuda' if t.cuda.is_available() else 'cpu'

#         self.to(self.device)

    
#     def forward(self, state):
#         prob = self.fc1(state)
#         prob = F.relu(prob)
#         prob = self.fc2(prob)
#         prob = F.relu(prob)
#         mu = self.mu(prob)
#         sigma = self.sigma(prob)

#         sigma = t.clamp(sigma, min=self.reparam_noise, max=1)
        
#         return mu, sigma
    
#     def sample_normal(self, state, reparameterize=True):
#         mu, sigma = self.forward(state)
#         print('\n\n\nthis is mu and sigma : ', mu, sigma)
#         probabilities = Normal(mu, sigma)

#         if reparameterize:
#             actions = probabilities.rsample()
#         else:
#             actions = probabilities.sample()

#         action = t.tanh(actions) * t.tensor(self.max_action).to(self.device)
#         log_probs = probabilities.log_prob(actions)
#         log_probs -= t.log(1 - action.pow(2) + self.reparam_noise)
#         log_probs = log_probs.sum(1, keepdim=True)
#         # print('\n\n',action, log_probs)
#         return action, log_probs
    


# class ValueNetwork(nn.Module):
#     def __init__(self, beta, input_dims, output_dims, fc1_dims, fc2_dims):
#         super(ValueNetwork, self).__init__()

#         self.input_dims = input_dims
#         self.output_dims = output_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
        

#         self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.v = nn.Linear(self.fc2_dims, 1)

#         self.optimizer = optim.Adam(self.parameters(), lr=beta)
#         self.device = 'cuda' if t.cuda.is_available() else 'cpu'

#         self.to(self.device)


#     def forward(self, state):
#         state_value = self.fc1(state)
#         state_value = F.relu(state_value)
#         state_value = self.fc2(state_value)
#         state_value = F.relu(state_value)

#         v = self.v(state_value)
#         return v
    


# class Agent:
#     def __init__(self, alpha, beta, input_dims, n_actions, tau, env, gamma=0.99, max_size=100000, 
#                  fc1_dims=256, fc2_dims=256, batch_size=100, reward_scale=2):
#         print(env.action_space.high)
#         self.gamma = gamma
#         self.tau = tau
#         self.memory = Memory(batch_size, max_size, env, 'cuda')
#         self.batch_size = batch_size
#         self.n_actions = n_actions

#         self.actor = ActorNetwork(alpha, input_dims, n_actions, fc1_dims, fc2_dims, env.action_space.high)
#         self.critic1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
#         self.critic2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
#         self.value = ValueNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims)
#         self.target_value = ValueNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims)
        
#         self.scale = reward_scale
#         self.update_network_parameters(tau=1)

#         self.device = 'cuda' if t.cuda.is_available() else 'cpu'
    
#     def choose_action(self, state):
#         state = t.Tensor([state]).to(self.device)
#         with t.no_grad():
#             actions, _ = self.actor.sample_normal(state, reparameterize=False)
#         return actions.cpu().detach().numpy()[0]
    
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.store_memory(state, action, reward, next_state, done)
    
#     def update_network_parameters(self, tau=None):
#         if tau is None:
#             tau = self.tau
        
#         target_value_params = self.target_value.named_parameters()
#         value_params = self.value.named_parameters()
        
#         target_value_state_dict = dict(target_value_params)
#         value_state_dict = dict(value_params)

#         for name in value_state_dict:
#             value_state_dict[name] = tau * value_state_dict[name].clone() + \
#                                     (1-tau)*target_value_state_dict[name].clone()
        
#         self.target_value.load_state_dict(value_state_dict)
    

#     def learn(self):
#         if self.memory.mem_ctr < self.batch_size:
#             return
        
#         state, action, reward, next_state, done = self.memory.generate_batches()

        
#         value = self.value(state).view(-1)
#         value_ = self.target_value(next_state).view(-1)
#         value_[done] = 0.0

#         actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
#         log_probs = log_probs.view(-1)
#         q1_new_policy = self.critic1.forward(state, actions)
#         q2_new_policy = self.critic2.forward(state, actions)
#         critic_value = t.min(q1_new_policy, q2_new_policy)
#         critic_value = critic_value.view(-1)

#         self.value.optimizer.zero_grad()
#         value_target = critic_value - log_probs
#         value_loss = 0.5 * F.mse_loss(value, value_target)
#         value_loss.backward(retain_graph=True)
#         self.value.optimizer.step()

#         actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
#         log_probs = log_probs.view(-1)
#         q1_new_policy = self.critic1.forward(state, actions)
#         q2_new_policy = self.critic2.forward(state, actions)
#         critic_value = t.min(q1_new_policy, q2_new_policy)
#         critic_value = critic_value.view(-1)

#         actor_loss = log_probs - critic_value
#         actor_loss = t.mean(actor_loss)
#         self.actor.optimizer.zero_grad()
#         actor_loss.backward(retain_graph=True)
#         self.actor.optimizer.step()

#         q_hat = self.scale * reward + self.gamma * value_
#         q1_old_policy = self.critic1.forward(state, action).view(-1)
#         q2_old_policy = self.critic2.forward(state, action).view(-1)
#         critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
#         critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        
#         self.critic1.optimizer.zero_grad()
#         self.critic2.optimizer.zero_grad()
#         critic_loss = critic_1_loss + critic_2_loss
#         critic_loss.backward()
#         self.critic1.optimizer.step()
#         self.critic2.optimizer.step()

        
#         self.update_network_parameters()

# def run():
#     env_id = 'InvertedPendulum-v2'
#     env = gym.make(env_id)

#     input_dims = np.array(env.observation_space.shape).prod()
#     n_actions = np.array(env.action_space.shape).prod()

#     agent = Agent(alpha=0.0003, beta=0.0003, tau=0.005,
#                   reward_scale=2, env=env,
#                    input_dims=input_dims, n_actions=n_actions, batch_size=256)
    
#     best_score = env.reward_range[0]
#     score_history = []

#     steps = 0
#     n_games = 250
#     for i in range(n_games):
#         score = 0
#         done = False
#         state = env.reset()
#         while not done:
#             action = agent.choose_action(state)
#             next_state, reward, done, info = env.step(action)
#             steps += 1
#             score += reward
#             agent.remember(t.tensor(state), t.tensor(action), reward, t.tensor(next_state), done)
#             agent.learn()
#             state = next_state

#         score_history.append(score)
#         avg_score = np.mean(score_history[-20:])

#         print(f'episode : {i}, score : {score}, avg_score : {avg_score}')
        



# if __name__=='__main__':
#     run()  







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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 ):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        
        self.to(self.device)
    
    def forward(self, state, action):
        action_value = self.fc1(t.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q1 = self.q1(action_value)
        return q1
    

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims, fc2_dims, max_action):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.reparam_noise = 1e-6
        self.max_action = max_action
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'

        self.to(self.device)

    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = t.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)

        if reparameterize:
            u = dist.rsample()
        else:
            u = dist.sample()

        # squashed action
        a = t.tanh(u)
        action = a * t.tensor(self.max_action).to(self.device)

        # log prob correction
        log_prob = dist.log_prob(u)
        log_prob -= t.log(1 - a.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, output_dims, fc1_dims, fc2_dims):
        super(ValueNetwork, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'

        self.to(self.device)


    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v
    


class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions, tau, env, gamma=0.99, max_size=100000, 
                 fc1_dims=256, fc2_dims=256, batch_size=100, reward_scale=2):
        print(env.action_space.high)
        self.gamma = gamma
        self.tau = tau
        self.memory = Memory(batch_size, max_size, env, 'cuda')
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions, fc1_dims, fc2_dims, env.action_space.high)
        self.critic1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
        self.critic2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions)
        self.value = ValueNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims)
        self.target_value = ValueNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims)
        
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.alpha = 0.2

        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
    
    def choose_action(self, state):
        state = t.Tensor([state]).to(self.device)
        with t.no_grad():
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                    (1-tau)*target_value_state_dict[name].clone()
        
        self.target_value.load_state_dict(value_state_dict)
    

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.generate_batches()

        
        value = self.value(state).view(-1)
        value_ = self.target_value(next_state).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1 = self.critic1(state, actions).view(-1)
        q2 = self.critic2(state, actions).view(-1)
        min_q = t.min(q1, q2)

        value_target = (min_q - self.alpha * log_probs).detach()

        self.value.optimizer.zero_grad()
        value_loss = F.mse_loss(value, value_target)
        value_loss.backward()
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1 = self.critic1(state, actions).view(-1)
        q2 = self.critic2(state, actions).view(-1)
        min_q = t.min(q1, q2)

        actor_loss = (self.alpha * log_probs - min_q).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        q_hat = reward * self.scale + self.gamma * value_ * (~done)

        q1_old = self.critic1(state, action).view(-1)
        q2_old = self.critic2(state, action).view(-1)

        critic1_loss = F.mse_loss(q1_old, q_hat)
        critic2_loss = F.mse_loss(q2_old, q_hat)
        critic_loss = critic1_loss + critic2_loss

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        
        self.update_network_parameters()

def run():
    env_id = 'InvertedPendulum-v2'
    env = gym.make(env_id)

    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = np.array(env.action_space.shape).prod()

    agent = Agent(alpha=0.0003, beta=0.0003, tau=0.005,
                  reward_scale=2, env=env,
                   input_dims=input_dims, n_actions=n_actions, batch_size=256)
    
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






    