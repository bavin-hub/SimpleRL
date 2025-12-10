import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch.optim as optim
import gym


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
        self.v = nn.Linear(self.fc1_dims, 1)
        self.a = nn.Linear(self.fc1_dims, self.output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.td_loss = nn.MSELoss()

        device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(device)


    def forward(self, state):
        hidden_states = F.relu(self.fc1(state))
        v = self.v(hidden_states)
        a = self.a(hidden_states)
        return v, a
    


class Agent:
    def __init__(self, env, gamma, epsilon, lr, input_dims, n_actions, 
                 memory_size, batch_size, eps_end=0.01, eps_dec=5e-4,
                 fc1_dims=512, fc2_dims=256):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.input_dims = input_dims
        # print(type(n_actions))
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size

        self.replace_target_cnt = 100
        self.learn_step_counter = 0

        self.memory = Memory(batch_size, memory_size, env, 'cuda')

        self.q_network = DeepQNetwork(lr, input_dims, n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.q_next_network = DeepQNetwork(lr, input_dims, n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = t.tensor([state]).to(self.device)
            _, actions = self.q_network.forward(state)
            action = t.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def replace_targe_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next_network.load_state_dict(self.q_network.state_dict())         
    
    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state, action, reward, next_state, done = self.memory.generate_batches()
    
        self.replace_targe_network()
        
        v_s, a_s = self.q_network.forward(state)
        v_s_, a_s_ = self.q_next_network.forward(next_state)

        v_s_eval, a_s_eval = self.q_network.forward(next_state)

        q_pred = t.add(v_s, (a_s - a_s.mean(dim=1, keepdim=True)))[batch_index, action]
        q_next = t.add(v_s_, (a_s_ - a_s_.mean(dim=1, keepdims=True)))
        q_eval = t.add(v_s_eval, (a_s_eval - a_s_eval.mean(dim=1, keepdims=True)))
        
        max_actions = t.argmax(q_eval, dim=1)

        q_next[done] = 0.0
        q_target = reward + self.gamma * q_next[batch_index, max_actions]

        self.q_network.optimizer.zero_grad()
        loss = self.q_network.td_loss(q_target, q_pred)
        loss.backward()
        self.q_network.optimizer.step()
        self.learn_step_counter += 1


        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
        

def run():
    env = gym.make('LunarLander-v2')
    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = env.action_space.n

    agent = Agent(env, gamma=0.99, epsilon=1.0, lr=5e-4,
                  input_dims=input_dims, n_actions=n_actions, memory_size=100000, batch_size=64)

    n_games = 500
    score_history = []

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


if __name__=='__main__':
    run()
        


        
        

        

        
        


