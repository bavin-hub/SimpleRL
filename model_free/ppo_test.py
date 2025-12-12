import os
import gym

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--eval_episodes', type=int, default=1)

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

        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.model = Network(np.array(env.observation_space.shape).prod(), env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)   

        self.memory = Memory(self.batch_size)


        # save and load models
        cwd = os.getcwd()
        self.saved_models_path = os.path.join(cwd, 'saved_model_wts')
        if os.path.isdir(self.saved_models_path):
            pass
        else:
            print('creating saved models dir to store weights')
            os.makedirs(self.saved_models_path)



    def remember(self, state, action, reward, value, logprob, done):
        self.memory.store_memory(state,
                                 action,
                                 reward,
                                 value,
                                 logprob,
                                 done)
        
    def take_action(self, state):
        state = t.tensor([state], dtype=t.float).to(self.device)
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
                mb_states = t.tensor(b_states[batch], dtype=t.float).to(self.device)
                mb_old_logprobs = t.tensor(b_old_logprobs[batch], dtype=t.float).to(self.device)
                mb_actions = t.tensor(b_actions[batch], dtype=t.float).to(self.device)
                mb_advantages = t.tensor(b_advantages[batch], dtype=t.float).to(self.device)
                mb_returns = t.tensor(b_returns[batch], dtype=t.float).to(self.device)

                _, mb_new_logprobs, new_values, entropy = self.model.get_action_value(mb_states, mb_actions)
                
                # actor loss
                ratio = (mb_new_logprobs - mb_old_logprobs).exp()
                actor_loss1 = mb_advantages * ratio
                actor_loss2 = mb_advantages * t.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -t.min(actor_loss1, actor_loss2).mean()

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

    
    def save_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        t.save(self.model.actor.state_dict(), model_path)
        print('saved model successfully')


    def load_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        if os.path.isfile(model_path):
            self.model.actor.load_state_dict(t.load(model_path, weights_only=True))
            print('model loaded successfully')
        else:
            print('\n\nWeights does not exists!! Train the model first')

    def deteministic_action(self, state):
        self.model.actor.eval()
        state = t.tensor([state], dtype=t.float).to('cuda')
        probs = self.model.actor.forward(state).squeeze()
        action = t.argmax(probs)
        print('this is action : ', action)
        return action.cpu().detach().numpy()
    
    def save_video(self, episode_frames, env_name, ep_num=1):
        from utils import save_gif
        save_gif(episode_frames, env_name, ep_num)



def run(args):
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='rgb_array')
    num_episodes = num_iterations = 2
    episode_len = steps_per_episode = 256
    
    agent = Agent(env, gamma=0.99, alpha=2.5e-4, gae_l=0.96, policy_clip=0.2, 
                  batch_size=64, n_epochs=4)

    state = env.reset()
    score_history = []
    avg_score = 0

    if args.run_mode == 'train':

        for ep in range(1, num_episodes + 1):
            episode_score = 0
            print(f'Simulating episode : {ep}')
            with t.no_grad():
                for step in range(steps_per_episode):
                    
                    action, logprob, value = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    agent.remember(state, action, reward, value, logprob, done)
                    episode_score += reward
                    state = next_state

                    # print(reward)
                    if done:
                        state = env.reset()

            agent.learn()

            score_history.append(episode_score)
            # print(score_history)
            avg_score = np.mean(score_history[-100:])
            print(f'ep-{ep} ; avg_score - {avg_score}\n\n')

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


if __name__=='__main__':
    args = parser.parse_args()
    run(args)