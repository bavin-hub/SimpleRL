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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--eval_episodes', type=int, default=1)




class Memory:
    def __init__(self, batch_size, memory_size, env, device):
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_shape = np.array(env.observation_space.shape).prod()
        self.action_shape = np.array(env.action_space.shape).prod()
        self.initialize_memory()
    
    def initialize_memory(self):
        self.states = t.zeros(self.memory_size, self.state_shape, device=self.device)
        self.actions = t.zeros(self.memory_size, self.action_shape, device=self.device)
        self.rewards = t.zeros(self.memory_size, 1, device=self.device)
        self.values = t.zeros(self.memory_size, 1, device=self.device)
        self.logprobs = t.zeros(self.memory_size, 1, device=self.device)
        self.dones = t.zeros(self.memory_size, 1, device=self.device)


    def generate_batches(self):
        total_samples = self.memory_size
        n_mini_batches = np.arange(0, total_samples, self.batch_size)
        samples_indices = np.arange(total_samples)
        np.random.shuffle(samples_indices)
        mini_batch_indices = [samples_indices[i:i+self.batch_size] for i in n_mini_batches]
        return self.states, self.actions, self.rewards,\
              self.values, self.logprobs, self.dones, mini_batch_indices   

    def store_memory(self, state, action, reward, value, logprob, done, idx):
        idx = idx % self.memory_size
        # if idx < self.memory_size:
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.logprobs[idx] = logprob
        self.dones[idx] = int(done)

        
    def clear_memory(self):
        # self.free_memory()
        self.initialize_memory()


    def free_memory(self):
        del self.states, self.actions, self.rewards,\
            self.values, self.logprobs, self.dones
        t.cuda.empty_cache()
        





class Network(nn.Module):
    def __init__(self, env, hidden, std_range, std_init):
        super(Network, self).__init__()
        self.input_dims = np.array(env.observation_space.shape).prod()
        self.output_dims = np.array(env.action_space.shape).prod()
        self.hidden = hidden

        self.actor = self.init_actor()
        self.critic = self.init_critic()

        self.logstd_range = (np.log(std_range[0]), np.log(std_range[1]))
        self.logstd_net = nn.Parameter(t.full((1,self.output_dims), np.log(std_init), dtype=t.float32))
        # self.logstd_net = nn.Parameter(t.zeros(1, self.output_dims))
        # self.logstd_net = nn.Parameter(t.ones(1, self.output_dims) * -0.5)
    
    def layer_mod(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal(layer.weight, std)
        nn.init.constant(layer.bias, bias_const)
        return layer
    
    def init_actor(self):
        return nn.Sequential(
            nn.Linear(self.input_dims, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.output_dims),
            nn.Tanh()
        )
    
    def init_critic(self):
        return nn.Sequential(
            nn.Linear(self.input_dims, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),

        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_value(self, x, action=None, deteministic=False):
        mean_logits = self.actor(x)

        if deteministic:
            # print('deterministic action shape : ', mean_logits.shape)
            return mean_logits
            # return mean_logits

        self.logstd_net.data.clip(*self.logstd_range)
        std_logits = self.logstd_net.exp().expand_as(mean_logits)

        probs = Normal(mean_logits, std_logits)
        if action is None:
            action = probs.sample()

        return action.clip(-1.0, 1.0), self.critic(x), probs.log_prob(action).sum(-1), probs.entropy().sum(-1)


class Agent():
    def __init__(self, env, batch_size, memory_size, hidden, epochs, lr, gamma, gae_l,\
                 actor_clip_coeff, value_clip_coeff, entropy_clip_coeff, device):
        
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.gae_l = gae_l
        self.actor_clip_coeff = actor_clip_coeff
        self.value_clip_coeff = value_clip_coeff
        self.entropy_clip_coeff = entropy_clip_coeff

        self.memory = Memory(batch_size, memory_size, env, self.device)
        self.network = Network(env, hidden, std_range=(0.03, 0.5), std_init=0.4).to(device)
        # print('\n\n',self.network.parameters,'\n\n')
        self.optimizer = optim.Adam(self.network.parameters(), 
                                    lr=lr)

        self.advantages = t.zeros(memory_size, 1, dtype=t.float, device=self.device)
        self.returns = t.zeros(memory_size, 1, dtype=t.float, device=self.device)

        self.critic_loss_func = t.nn.MSELoss()


        # save and load models
        cwd = os.getcwd()
        self.saved_models_path = os.path.join(cwd, 'saved_model_wts')
        if os.path.isdir(self.saved_models_path):
            pass
        else:
            print('creating saved models dir to store weights')
            os.makedirs(self.saved_models_path)

    def store(self, state, action, reward, value, logprob, done, idx):
        self.memory.store_memory(state, action, reward,\
                                 value, logprob, done, idx)    

    def sample_action(self, x, deterministic):
        with t.no_grad():
            x = t.tensor(x, device=self.device, dtype=t.float).unsqueeze(0)
            action, value, logprob, _ = self.network.get_action_value(x, action=None, deteministic=deterministic)
        return action, value, logprob
    
    def compute_gae(self, rewards, values, dones, normalize=True):
        # print(dones)
        def reward_normalization(rewards):
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        rewards = reward_normalization(rewards)

        # not mine
        returns = t.zeros_like(rewards)
        advantages = t.zeros_like(rewards)
        masks = (1-dones) * 1
        running_returns = 0
        previous_value = 0
        running_adv = 0
        for k in reversed(range(0, len(rewards))):
            running_returns = rewards[k] + self.gamma * running_returns * masks[k]
            running_td_error = rewards[k] + self.gamma * previous_value * masks[k] - values[k]
            running_adv = running_td_error + self.gamma * self.gae_l * running_adv * masks[k]
            returns[k] = running_adv + values[k]
            previous_value = values[k]
            advantages[k] = running_adv


        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-3)
        
        # returns = advantages + values
        
        return advantages, returns

    def learn(self):
        states, actions, rewards,\
        values, old_logprobs, dones, batch_indices = self.memory.generate_batches()

        # compute gae
        advantages, returns = self.compute_gae(rewards, values, dones)

        print('Learning started')
        # run training loop
        for epoch in range(self.epochs):
            for mb_indices in batch_indices:
                # print(mb_indices)
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices].squeeze(-1)
                mb_returns = returns[mb_indices]
                # print('while training  ', mb_actions.shape)
                _, new_values, new_logprobs, entropy = self.network.get_action_value(mb_states, mb_actions, deteministic=False)
        
                # actor loss
                ratio = t.exp(new_logprobs.squeeze(-1) - mb_old_log_probs.squeeze(-1))
                surr1 =  ratio * mb_advantages
                surr2 =  t.clamp(ratio, 1-self.actor_clip_coeff, 1+self.actor_clip_coeff) * mb_advantages
                actor_loss = -t.min(surr1, surr2).mean()

                # entopy loss
                entropy_loss = self.entropy_clip_coeff * entropy.squeeze(-1).mean()

                # critic loss
                critic_loss = F.smooth_l1_loss(new_values, mb_returns)

                loss = actor_loss + critic_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
        
        self.memory.clear_memory()
        print('Learning completed')

    def save_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        t.save(self.network.actor.state_dict(), model_path)
        print('saved model successfully')


    def load_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        if os.path.isfile(model_path):
            self.network.actor.load_state_dict(t.load(model_path, weights_only=True))
            print('model loaded successfully')
        else:
            print('\n\nWeights does not exists!! Train the model first')

    def deteministic_action(self, state):
        self.network.eval()
        state = t.tensor([state], dtype=t.float).to('cuda')
        action = self.network.actor.forward(state)
        return action.cpu().detach().numpy()[0] 
    
    def save_video(self, episode_frames, env_name, ep_num=1):
        from utils import save_gif
        save_gif(episode_frames, env_name, ep_num)




class Nomalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean

        x = x / (self.std + 1e-8)

        x = np.clip(x, -5, +5)


        return x
    



def evaluate(eval_env, agent, eval_num, deterministic=True):
    reward_list = []
    env = eval_env

    normalize = Nomalize(env.observation_space.shape[0])

    def get_state_tensor(state):
        return t.tensor(state, dtype=t.float32).unsqueeze(0)
    
    for _ in range(eval_num):
        state = normalize(env.reset())
        state = get_state_tensor(state)

        total_reward = 0.0

        for step in range(1000):
            with t.no_grad():
                action, _, _ = agent.sample_action(state, deterministic=deterministic)
                # print(action)
            state, reward, done, _ = env.step(action.to('cpu').numpy().squeeze())
            state = get_state_tensor(normalize(state))

            total_reward += reward

            if done:
                break
        
        reward_list.append(total_reward)

    
    mean = np.array(reward_list).mean()
    return mean





def run(args):

    env_name = 'Hopper-v2'
    env = gym.make(env_name, render_mode='rgb_array')

    eval_env = gym.make(env_name)
    eval_num = 4

    nomalize = Nomalize(env.observation_space.shape[0])

    batch_size = 64
    memory_size = 2048
    epochs = 15

    hidden = 128
    lr = 1e-4  
    gamma = 0.99
    gae = 0.95
    
    actor_clip_coeff = 0.2
    value_clip_coeff = 0.5
    entropy_clip_coeff = 0.001

    

    agent = Agent(env, batch_size, memory_size, hidden,\
                  epochs, lr, gamma, gae,\
                   actor_clip_coeff, value_clip_coeff, entropy_clip_coeff, 'cuda')
    
    num_episodes = 10000000
    rollout_len = episode_len = memory_size
    total_steps = 10

    score_history = []
    step = 0

    if args.run_mode == 'train':
        state = nomalize(env.reset())
        for step in range(total_steps):
            with t.no_grad():
                action, value, logprob = agent.sample_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action.to('cpu').numpy().squeeze())
            step += 1
            agent.store(t.tensor(state, device='cuda'), action, reward, value, logprob, done, step)
            
            state = nomalize(next_state)
            if done:
                state = nomalize(env.reset())


            if step % memory_size == 0:
                print('\n\nLearning Phase')
                agent.learn()
                print('Evaluation Phase')
                mean_eval_returns = evaluate(eval_env, agent, eval_num, deterministic=False)
                print(f'step : {step} eval_avg_rets : {mean_eval_returns}')

        agent.save_model(env_name)

    elif args.run_mode == 'eval':
        eval_episodes = args.eval_episodes
        agent.load_model(env_name)

        for eval_ep in range(eval_episodes):
            state = env.reset()
            done = False
            score = 0
            episode_frames = []
            while not done:
                frame = env.render()
                frame = np.array(frame).squeeze()
                episode_frames.append(frame)

                action = agent.deteministic_action(state)
                next_state, reward, done, info = env.step(action)
                score += reward
                state = next_state

            print(f'eval ep: {eval_ep}, score: {score}')

            agent.save_video(episode_frames, env_name, ep_num=eval_ep)

    env.close()

if __name__=='__main__':
    args = parser.parse_args()
    run(args)

