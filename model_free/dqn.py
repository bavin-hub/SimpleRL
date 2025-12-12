import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch.optim as optim
import gym
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--eval_episodes', type=int, default=1)



class Memory:
    def __init__(self, batch_size, memory_size, env, device):
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.mem_ctr = 0
        self.state_shape = np.array(env.observation_space.shape).prod()
        self.action_shape = env.action_space.n
        self.initialize_memory()
    
    def initialize_memory(self):
        self.states = t.zeros(self.memory_size, self.state_shape, device=self.device)
        self.actions = t.zeros(self.memory_size, device=self.device, dtype=t.int32)
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




class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q_ = nn.Linear(self.fc2_dims, self.output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.td_loss = nn.MSELoss()

        device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.q_(x)

        return q_values
    


class Agent:
    def __init__(self, env, gamma, epsilon, lr, input_dims, n_actions, 
                 memory_size, batch_size, eps_end=0.01, eps_dec=5e-4,
                 fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.input_dims = input_dims
        # print(type(n_actions))
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size

        self.memory = Memory(batch_size, memory_size, env, 'cuda')

        self.q_network = DeepQNetwork(lr, input_dims, n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'

        # save and load models
        cwd = os.getcwd()
        self.saved_models_path = os.path.join(cwd, 'saved_model_wts')
        if os.path.isdir(self.saved_models_path):
            pass
        else:
            print('creating saved models dir to store weights')
            os.makedirs(self.saved_models_path)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = t.tensor([state]).to(self.device)
            actions = self.q_network.forward(state)
            action = t.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state, action, reward, next_state, done = self.memory.generate_batches()
    
        q_value = self.q_network.forward(state)[batch_index, action]

        q_value_ = self.q_network.forward(next_state)
        q_value_[done] = 0.0

        y_target = reward + self.gamma * t.max(q_value_, dim=1)[0]
        
        self.q_network.optimizer.zero_grad()
        loss = self.q_network.td_loss(y_target, q_value)
        loss.backward()
        self.q_network.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
        

    def save_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        t.save(self.q_network.state_dict(), model_path)
        print('saved model successfully')


    def load_model(self, env_name):
        model_path = f'{self.saved_models_path}/{env_name}_wts.pt'
        if os.path.isfile(model_path):
            self.q_network.load_state_dict(t.load(model_path, weights_only=True))
            print('model loaded successfully')
        else:
            print('\n\nWeights does not exists!! Train the model first')

    def deteministic_action(self, state):
        self.q_network.eval()
        state = t.tensor([state], dtype=t.float).to('cuda')
        q_values = self.q_network.forward(state).squeeze()
        action = t.argmax(q_values)
        return action.cpu().detach().numpy() 
    
    def save_video(self, episode_frames, env_name, ep_num=1):
        from utils import save_gif
        save_gif(episode_frames, env_name, ep_num)
        

def run(args):
    env_name = 'LunarLander'
    env = gym.make(env_name, render_mode='rgb_array')
    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = env.action_space.n

    agent = Agent(env, gamma=0.99, epsilon=1.0, lr=3e-3,
                  input_dims=input_dims, n_actions=n_actions, memory_size=100000, batch_size=64)

    n_games = 2
    score_history = []

    if args.run_mode == 'train':
        for i in range(n_games):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(t.tensor(state), action, reward, t.tensor(next_state), done)

                agent.learn()
                state = next_state

            score_history.append(score)
            avg_score = np.mean(score_history[-20:])

            print(f'episode : {i}, score : {score}, avg_score : {avg_score}')
        
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
        


        
        

        

        
        


