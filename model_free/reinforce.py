import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch.optim as optim
import gym
from torch.distributions import Categorical
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--eval_episodes', type=int, default=1)


class PGAgent(nn.Module):
    def __init__(self, input_dims, output_dims, lr, device, dense_dims=256):
        super(PGAgent, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.device = device
        self.lr = lr

        self.fc1 = nn.Linear(self.input_dims, dense_dims)
        self.fc2 = nn.Linear(dense_dims, dense_dims)
        self.fc3 = nn.Linear(dense_dims, self.output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        outputs = F.softmax(self.fc3(x), dim=-1)
        return outputs
    


class Agent:
    def __init__(self, gamma, input_dims, output_dims, lr, device):
        self.gamma = gamma
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        self.device = device

        self.pg_net = PGAgent(self.input_dims, self.output_dims, self.lr, self.device)

        self.states = []
        self.rewards = []
        self.actions = []

        # save and load models
        cwd = os.getcwd()
        self.saved_models_path = os.path.join(cwd, 'saved_model_wts')
        if os.path.isdir(self.saved_models_path):
            pass
        else:
            print('creating saved models dir to store weights')
            os.makedirs(self.saved_models_path)
    
    def choose_action(self, state):
        with t.no_grad():
            state = t.tensor([state], dtype=t.float32).to(self.device)
            probs = self.pg_net.forward(state)
            dist = Categorical(probs)
            action = dist.sample()
        return action.cpu().detach().numpy()[0]
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def compute_discounted_returns(self, rewards):
        # print(len(rewards))
        G = t.zeros(len(rewards)).to(self.device)
        for i in range(len(rewards)):
            Gt = 0
            discount = 1
            for r in range(i, len(rewards)):
                Gt += rewards[r] * discount
                discount *= self.gamma
            G[i] = Gt

        G = (G - G.mean()) / (G.std() + 1e-8)
        
        return G


    def learn(self):

        returns = self.compute_discounted_returns(self.rewards)
        
        # convert to tensors and move to gpu
        states = t.tensor(np.array(self.states), dtype=t.float32).to(self.device)
        actions = t.tensor(self.actions, dtype=t.long).to(self.device)


        # training
        probs = self.pg_net.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)        
        
        
        pg_loss = -(log_probs * returns).mean()
        self.pg_net.optimizer.zero_grad()
        pg_loss.backward()
        self.pg_net.optimizer.step()

        self.clear_memory()

    
    def save_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        t.save(self.pg_net.state_dict(), model_path)
        print('saved model successfully')


    def load_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        if os.path.isfile(model_path):
            self.pg_net.load_state_dict(t.load(model_path, weights_only=True))
            print('model loaded successfully')
        else:
            print('\n\nWeights does not exists!! Train the model first')

    def deteministic_action(self, state):
        self.pg_net.eval()
        state = t.tensor([state], dtype=t.float).to('cuda')
        probs = self.pg_net.forward(state).squeeze()
        action = t.argmax(probs)
        # print('this is action : ', action)
        return action.cpu().detach().numpy()
    
    def save_video(self, episode_frames, env_name, ep_num=1):
        from utils import save_gif
        save_gif(episode_frames, env_name, ep_num)




def run(args):
    env_name = 'CartPole'
    env = gym.make(env_name, render_mode='rgb_array')
    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = env.action_space.n
    gamma = 0.99
    lr = 1e-4
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    agent = Agent(gamma=gamma, input_dims=input_dims, output_dims=n_actions,
                  lr=lr, device=device) 
    
    n_episodes = 2
    all_episode_scores = []

    if args.run_mode == 'train':

        for ep in range(1, n_episodes + 1):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _= env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                score += reward
            
            all_episode_scores.append(score)

            agent.learn()

            avg_score = np.mean(all_episode_scores[-20:])
            print(f'ep: {ep}, score: {score}, avg_score: {avg_score}')
            
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


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)