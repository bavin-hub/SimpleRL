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


class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_pred = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



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

        
    # def clear_memory(self):
    #     # self.free_memory()
    #     self.initialize_memory()


    # def free_memory(self):
    #     del self.states, self.actions, self.rewards,\
    #         self.values, self.logprobs, self.dones
    #     t.cuda.empty_cache()





class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1/np.sqrt(self.action_value.weight.data.size()[0]) 
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)


        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(self.device)

    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(t.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value
    


class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name


        self.fc1 = nn.Linear(input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x) 
        x = t.tanh(self.mu(x))

        return x
    



class Agent():
    def __init__(self, env, alpha, beta, tau, gamma=0.99, 
                 max_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        # self.n_actions = n_actions
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.input_dims = np.array(env.observation_space.shape).prod()
        self.n_actions = np.array(env.action_space.shape).prod()

        self.device = 'cuda' if t.cuda.is_available() else 'cpu'

        self.memory = Memory(batch_size=batch_size, memory_size=max_size, env=env, device=self.device)

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

        self.actor = ActorNet(alpha, self.input_dims, fc1_dims, fc2_dims, self.n_actions, 'actor')
        self.critic = CriticNet(beta, self.input_dims, fc1_dims, fc2_dims, self.n_actions, 'critic')

        self.target_actor = ActorNet(alpha, self.input_dims, fc1_dims, fc2_dims, self.n_actions, 'target_actor')
        self.target_critic = CriticNet(beta, self.input_dims, fc1_dims, fc2_dims, self.n_actions, 'target_critic')

        self.update_parameters(tau=1)


    def update_parameters(self, tau=None):
        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1-tau) * target_actor_state_dict[name].clone()
            
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                     (1-tau) * target_critic_state_dict[name].clone()
            
        
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)


    def choose_action(self, state):
        self.actor.eval()
        state = t.tensor([state], dtype=t.float).to(self.device)
        mu = self.actor.forward(state)
        mu_prime = mu + t.tensor(self.noise(), dtype=t.float).to(self.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]     


    def remember(self, state, action, reward, next_state, done):
        self.memory.store_memory(state,
                                 action,
                                 reward,
                                 next_state,
                                 done)


    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return 
        
        states, actions, rewards, next_states, dones = self.memory.generate_batches()

        target_actions = self.target_actor.forward(next_states)
        critic_values_ = self.target_critic.forward(next_states, target_actions)
        critic_values = self.critic.forward(states, actions)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)
        # print(dones.shape)

        # print(dones)
        critic_values_[dones] = 0.0
        critic_values_ = critic_values_.view(-1)
        yt = rewards + self.gamma * critic_values_
        # print(yt.shape)
        yt = yt.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(yt, critic_values) 
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = t.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_parameters()


    def save_model(self):
        path = f'/home/bavin/sem1/intro_robot_learning/my_rl/models/mine_ddpg.pt'
        t.save(self.actor.state_dict(), path)
        print('saved model successfully')




def run_agent():
    env = gym.make('LunarLanderContinuous-v2')

    agent = Agent(env, alpha=0.0001, beta=0.001, tau=0.001)

    n_games = 1000
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(t.tensor(state), t.tensor(action), reward, t.tensor(next_state), done)
            agent.learn()
            score += reward
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
    

        print('episode ', i, ' score : ', score, ' avg score : ', avg_score)

    agent.save_model()

    

if __name__=='__main__':
    run_agent()












