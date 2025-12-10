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


# class Actor(nn.Module):
#     def __init__(self, n_s, n_a, hidden):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(n_s, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.mu = nn.Linear(hidden, n_a)
#         self.mu.weight.data.mul_(0.1)
#         self.mu.bias.data.mul_(0.0)
#         self.log_sigma = nn.Linear(hidden, n_a)
    
#     def forward(self, x):
#         x = t.tanh(self.fc1(x))
#         x = t.tanh(self.fc2(x))
#         mu = self.mu(x)
#         log_sigma = self.log_sigma(x)
        
#         sigma = t.exp(log_sigma)
#         return mu, sigma
    
#     def get_dist(self, x):
#         mu, sigma = self.forward(x)
#         dist = Normal(mu, sigma)
#         return dist
    
# class Critic(nn.Module):
#     def __init__(self, n_s, hidden):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(n_s, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, 1)
#         self.fc3.weight.data.mul_(0.1)
#         self.fc3.bias.data.mul_(0.0)
    
#     def forward(self, x):
#         x = t.tanh(self.fc1(x))
#         x = t.tanh(self.fc2(x))
#         return self.fc3(x)
    
#     def get_value(self, x):
#         return self.forward(x)
    

# class Network():
#     def __init__(self, env, hidden):
#         self.input_dims = np.array(env.observation_space.shape).prod()
#         self.output_dims = np.array(env.action_space.shape).prod()
#         self.hidden = hidden
        
#         self.actor = Actor(self.input_dims, self.output_dims, hidden).to('cuda')
#         self.critic = Critic(self.input_dims, hidden).to('cuda')

#         lr_actor = 3e-4 
#         lr_critic = lr_actor
#         decay = 0.001
#         self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
#         self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=decay)

#     def get_action_value(self, x, action=None):
#         dist = self.actor.get_dist(x)
#         if action is None:
#             action = dist.sample()
#             # print('while sampling : ', action.shape)
#         log_prob = dist.log_prob(action).sum(1)
#         entropy = dist.entropy().sum(1)
#         value = self.critic.get_value(x)
#         return action, value, log_prob, entropy

    


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


        # advantages = t.zeros_like(rewards, device=self.device)
        # last_advantage = 0
        # last_value = 0

        # for k in reversed(range(len(rewards))):
        #     mask = 1.0 - dones[k]
        #     next_value = values[k + 1] if k < len(rewards) - 1 else 0
        #     delta = rewards[k] + self.gamma * next_value * mask - values[k]
        #     last_advantage = delta + self.gamma * self.gae_l * mask * last_advantage
        #     advantages[k] = last_advantage

        # returns = advantages + values

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

        # # mine
        # advantages = t.zeros_like(rewards)
        # next_gae_value = 0 
        # for k in reversed(range(len(rewards))):
        #     if k == len(rewards) - 1:
        #         next_state_value = 0
        #     else:
        #         next_state_value = values[k+1]
            
        #     delta = rewards[k] + self.gamma * next_state_value * (1-dones[k]) - values[k]
        #     advantage = next_gae_value = delta + self.gamma * self.gae_l * next_gae_value
        #     advantages[k] = advantage 

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

                # actor_loss = surr_loss - entropy_loss
                # self.network.actor_optim.zero_grad()
                # surr_loss.backward()
                # self.network.actor_optim.step()

                # critic loss
                critic_loss = F.smooth_l1_loss(new_values, mb_returns)
                # critic_loss = self.value_clip_coeff * ((mb_returns - new_values)**2).mean()
                # critic_loss = self.critic_loss_func(new_values.squeeze(-1),mb_returns.squeeze(-1))

                # self.network.critic_optim.zero_grad()
                # critic_loss.backward()
                # self.network.critic_optim.step()
                

                loss = actor_loss + critic_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
        
        self.memory.clear_memory()
        print('Learning completed')


    # def save_model(self, env_name, avg_score):
    #     path = f'/home/bavin/sem1/intro_robot_learning/my_rl/models/{env_name}-wts.pt'
    #     data = {'model_state_dict': self.network.actor.state_dict()}
    #     t.save(data, path)
    #     print('saved model successfully')

    def save_model(self):
        path = f'/home/bavin/sem1/intro_robot_learning/my_rl/models/mine_wts.pt'
        t.save(self.network.state_dict(), path)
        print('saved model successfully')




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





def run():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print('\n\n\nAvailable device : ', device)
    env_name = 'Hopper-v2'
    env = gym.make(env_name)
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
                   actor_clip_coeff, value_clip_coeff, entropy_clip_coeff, device)
    
    num_episodes = 10000000
    rollout_len = episode_len = memory_size
    total_steps = 100000

    score_history = []
    step = 0

    state = nomalize(env.reset())

    for step in range(total_steps):
        with t.no_grad():
            action, value, logprob = agent.sample_action(state, deterministic=False)
        next_state, reward, done, _ = env.step(action.to('cpu').numpy().squeeze())
        step += 1
        agent.store(t.tensor(state, device=device), action, reward, value, logprob, done, step)
        
        state = nomalize(next_state)
        if done:
            state = env.reset()

        if step % memory_size == 0:
            print('\n\nLearning Phase')
            agent.learn()
            print('Evaluation Phase')
            mean_eval_returns = evaluate(eval_env, agent, eval_num, deterministic=False)
            print(f'step : {step} eval_avg_rets : {mean_eval_returns}')

    agent.save_model()

    # for ep in range(num_episodes):
    #     episode_score = 0
    #     while True:
    #         action, value, logprob = agent.sample_action(state, deterministic=False)
    #         next_state, reward, done, _ = env.step(action.to('cpu').numpy().squeeze())
    #         step += 1
    #         episode_score += reward
    #         agent.store(t.tensor(state, device=device), action, reward, value, logprob, done, step)            
            
    #         state = next_state
    #         if done:
    #             if episode_score > 6000:
    #                 agent.save_model(env_name, avg_score)
    #             score_history.append(episode_score)
    #             avg_score = np.mean(score_history[-4:])
    #             # print(f'episode : {ep} score : {episode_score} avg-score : {avg_score} step : {step}')
    #             episode_score = 0
    #             state = nomalize(env.reset())
    #             break
            
        
    #     if step >= rollout_len:
    #         print(f'\n\n\n Learning Phase at ep {ep} \n\n\n')
    #         agent.learn()
    #         mean_eval_returns = evaluate(eval_env, agent, eval_num, deterministic=False)
    #         print(f'episode : {ep} eval_avg_rets : {mean_eval_returns} step : {step}')
    #         step = 0



    # for ep in range(num_episodes):
    #     episode_score = 0
    #     single_roll_out_score = 0
    #     single_roll_out_steps = 0
    #     with t.no_grad():
    #         for step in range(episode_len):
    #             action, value, logprob = agent.sample_action(state)
    #             next_state, reward, done, _ = env.step(action.to('cpu').numpy().squeeze())
    #             episode_score += reward
    #             single_roll_out_score += reward
    #             single_roll_out_steps += 1
    #             agent.store(t.tensor(state, device=device), action, reward, value, logprob, done, step)

    #             state = nomalize(next_state)
    #             if done:
    #                 print('its done - single roll out score : ', single_roll_out_score, 'single roll out steps : ', single_roll_out_steps)
    #                 state = nomalize(env.reset())
    #                 single_roll_out_steps = 0
    #                 single_roll_out_score = 0

    #     agent.learn()

    #     score_history.append(episode_score)
    #     avg_score = np.mean(score_history[-100:])
    #     print(f'ep-{ep} ; avg_score - {avg_score}\n\n') 

    #     if avg_score > 10000:
    #         agent.save_model(env_name, avg_score)
    #         break

if __name__=='__main__':
    run()




'''
1. separate actor and critic network with optimizers and lr decay
2. normalization - states, rewards, advantages
3. big memory but small batch size
4. tensors should not broadcast in learn function
5. actor network with 2 heads - mean and mu
6. value loss scaling, gradient clipping
'''

'''
1. actor net final activation tanh (not making a huge difference, oscillations are there in both the cases)
2. std dev clip range, std dev init (not initializing std dev gives shit results....it has the utmost impact)
3. clip std dev, sample, do not sum entropy and logprobs (summing log probs works as well, not clipping the std_dev give high returns immediately)
4. less entropy coeff (0.01 is inducing more oscillations and the model is unstable compared to 0.001)
5. clip grad (if not there oscillations are heavy and not stable)
6. smooth F1 loss for value loss
7. write a separate evaluate function (avg of 4 runs)
8. state, reward and advantage normlization (has a very huge effect)
'''











  # gae = 0
        # for step in reversed(range(len(rewards)-1)):
        #     delta = rewards[step] + self.gamma * (1 - dones[step]) * values[step + 1] - values[step]
        #     gae = delta + self.gamma * self.gae_l * (1 - dones[step]) * gae
        #     self.advantages[step] = gae
        #     self.returns[step] = self.advantages[step] + values[step]

        # if normalize:
        #     self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


        # next_state_gae = 0
        # for k in reversed(range(rewards.shape[0])):
        #     if k == rewards.shape[0] - 1:
        #         next_state_value = 0
        #     else:
        #         next_state_value = values[k+1]
        #     delta = rewards[k] + (1 - dones[k]) * self.gamma * next_state_value - values[k]
        #     advantage = next_state_gae =  delta + self.gae_l * self.gamma * next_state_gae
        #     ret = delta + values[k]
        #     self.advantages[k] = advantage
        #     self.returns[k] = ret

        # if normalize:
        #     self.advantages = (self.advantages - t.mean(self.advantages)) / (t.std(self.advantages) + 1e-8)

        # print('here')
        # for j in range(rewards.shape[0]-1):
        #     discount = 1
        #     a_t = 0
        #     print(j)
        #     for k in range(j, len(rewards)-1):
        #         a_t += discount * (rewards[k] + self.gamma * values[k+1] * (1-dones[k])) - values[k]
        #         discount *= self.gamma * self.gae_l
        #     self.advantages[j] = a_t

        # returns = self.advantages + values
        # print('end')