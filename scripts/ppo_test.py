import os
import gym

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.rewards),\
               np.array(self.values), np.array(self.logprobs), np.array(self.dones), batches

    def store_memory(self, state, action, reward, value, logprob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.logprobs.clear()
        self.dones.clear()



def layer_init(layer, std_dev=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal(layer.weight, std_dev)
    nn.init.constant(layer.bias, bias_const)
    return layer

class Network(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Network, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_shape), std_dev=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std_dev=1.0)
        )

    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_value(self, x, action=None):
        dist = self.actor(x)
        probs = Categorical(logits=dist)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), self.critic(x), probs.entropy()
    


class Agent:
    def __init__(self, env, gamma=0.99, alpha=0.0003, gae_l=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10, entropy_coeff=0.01):
        self.gamma = gamma
        self.gae_l = gae_l
        self.lr = alpha
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.entropy_coeff = entropy_coeff

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network(np.array(env.observation_space.shape).prod(), env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)   

        self.memory = Memory(self.batch_size)

    def remember(self, state, action, reward, value, logprob, done):
        self.memory.store_memory(state,
                                 action,
                                 reward,
                                 value,
                                 logprob,
                                 done)
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, logprob, value, entropy = self.model.get_action_value(state)
        return action.to('cpu').numpy()[0], logprob.to('cpu').numpy()[0], value.to('cpu').numpy()[0][0]

    def compute_gae(self, rewards, values, dones, normalize=True):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        next_gae_value = 0 
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_state_value = 0
            else:
                next_state_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_state_value * (1-int(dones[t])) - values[t]
            advantage = next_gae_value = delta + self.gamma * self.gae_l * next_gae_value
            advantages[t] = advantage    

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        return advantages


    def learn(self):
        b_states, b_actions, b_rewards, \
            b_values, b_old_logprobs, b_dones, mb_indices = self.memory.generate_batches()
        
        b_advantages = self.compute_gae(b_rewards, b_values, b_dones)
        b_returns = b_advantages + b_values

        for epoch in range(self.epochs): 
            for batch in mb_indices:  
                mb_states = torch.tensor(b_states[batch], dtype=torch.float).to(self.device)
                mb_old_logprobs = torch.tensor(b_old_logprobs[batch], dtype=torch.float).to(self.device)
                mb_actions = torch.tensor(b_actions[batch], dtype=torch.float).to(self.device)
                mb_advantages = torch.tensor(b_advantages[batch], dtype=torch.float).to(self.device)
                mb_returns = torch.tensor(b_returns[batch], dtype=torch.float).to(self.device)

                _, mb_new_logprobs, new_values, entropy = self.model.get_action_value(mb_states, mb_actions)
                
                # actor loss
                ratio = (mb_new_logprobs - mb_old_logprobs).exp()
                actor_loss1 = mb_advantages * ratio
                actor_loss2 = mb_advantages * torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(actor_loss1, actor_loss2).mean()

                # critic loss
                critic_loss = ((mb_returns - new_values)**2).mean()

                #entropy loss
                entropy_loss = entropy.mean()

                # final loss
                ppo_loss = actor_loss - self.entropy_coeff * entropy_loss + 0.5 * critic_loss

                # backprop and update params
                self.model.zero_grad()
                ppo_loss.backward()
                self.optimizer.step()
        
        self.memory.clear_memory()



def run():
    env = gym.make('CartPole-v1', render_mode='human')
    num_episodes = num_iterations = 200
    episode_len = steps_per_episode = 256
    
    agent = Agent(env, gamma=0.99, alpha=2.5e-4, gae_l=0.96, policy_clip=0.2, 
                  batch_size=64, n_epochs=4)

    state = env.reset()
    score_history = []
    avg_score = 0

    for ep in range(1, num_episodes + 1):
        episode_score = 0
        print(f'Simulating episode : {ep}')
        with torch.no_grad():
            for step in range(steps_per_episode):
                
                action, logprob, value = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)

                agent.remember(state, action, reward, value, logprob, done)
                episode_score += reward
                state = next_state

                # print(reward)
                if done:
                    print('done')
                    state = env.reset()

        agent.learn()

        score_history.append(episode_score)
        # print(score_history)
        avg_score = np.mean(score_history[-100:])
        print(f'ep-{ep} ; avg_score - {avg_score}\n\n')


if __name__=='__main__':
    run()