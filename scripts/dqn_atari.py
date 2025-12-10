import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch.optim as optim
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import cv2


class Memory:
    
    IMAGE_SIZE = 84
    FRAME_STACKS = 4

    def __init__(self, batch_size, memory_size, env, device):
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.mem_ctr = 0
        self.state_shape = np.array(env.observation_space.shape).prod()
        self.action_shape = env.action_space.n
        self.initialize_memory()
    
    def initialize_memory(self):
        self.states = t.zeros(self.memory_size, self.FRAME_STACKS, self.IMAGE_SIZE, self.IMAGE_SIZE, device=self.device)
        self.actions = t.zeros(self.memory_size, device=self.device, dtype=t.int32)
        self.rewards = t.zeros(self.memory_size, device=self.device)
        self.next_states = t.zeros(self.memory_size, self.FRAME_STACKS, self.IMAGE_SIZE, self.IMAGE_SIZE, device=self.device)
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
        

        self.fc1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.fc2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(2592, 256)
        self.q_ = nn.Linear(256, self.output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.td_loss = nn.MSELoss()

        device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.to(device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.fc3(x))
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

    def preprocess_frames(self, frames):
        # grayscale, resize, rescale
        def preprocess_single_frame(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (84, 84))
            frame = frame * 1/255.0
            return frame
        
        processed_frames = [preprocess_single_frame(frame) 
                            for frame in frames]

        return t.tensor(processed_frames, dtype=t.float32)


    def store_transition(self, state_frames, action, reward, next_state_frames, done):
        state_frames_processed = self.preprocess_frames(state_frames)
        next_state_frames_processed = self.preprocess_frames(next_state_frames)
        self.memory.store_memory(state_frames_processed, action, reward, next_state_frames_processed, done)

    def choose_action(self, state_frames):
        if np.random.random() > self.epsilon:
            
            state_frames_processed = t.unsqueeze(self.preprocess_frames(state_frames), dim=0).to(self.device)
            # print('choose action : ',state_frames_processed.shape)
            actions = self.q_network.forward(state_frames_processed)
            action = t.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state, action, reward, next_state, done = self.memory.generate_batches()

        # print('training is happening : ', state.shape)
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
        

def run():
    env = gym.make('ALE/Breakout-v5')
    input_dims = np.array(env.observation_space.shape).prod()
    n_actions = env.action_space.n

    agent = Agent(env, gamma=0.99, epsilon=1.0, lr=3e-3,
                  input_dims=input_dims, n_actions=n_actions, memory_size=1000, batch_size=64)

    n_games = 500
    score_history = []
    frame_skip = 4
    sample_curr_frames = []
    sample_next_frames = []

    # initializing buffer for 4 frames
    state, _ = env.reset()
    for _ in range(frame_skip):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        sample_curr_frames.append(state)
        sample_next_frames.append(next_state)
    agent.store_transition(sample_curr_frames, action, reward, sample_next_frames, done)
    sample_curr_frames.clear()
    sample_next_frames.clear()

    print('after\n\n')
    for i in range(n_games):
        score = 0
        done = False
        state, _ = env.reset()
        skip_ctr = 0
        while not done:
            # if skip_ctr % frame_skip == 0:
            #     action = agent.choose_action(sample_curr_frames)
            next_state, reward, done, truncated, info = env.step(action)
            sample_curr_frames.append(state)
            sample_next_frames.append(next_state)
            score += reward
            skip_ctr += 1

            if skip_ctr % frame_skip == 0:
                # print(len(sample_curr_frames))
                agent.store_transition(sample_curr_frames[:4], action, reward, sample_next_frames[:4], done)
                action = agent.choose_action(sample_curr_frames[:4])
                sample_curr_frames.clear()
                sample_next_frames.clear()
            
            state = next_state

            agent.learn()

        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        print(f'episode : {i}, score : {score}, avg_score : {avg_score}')


if __name__=='__main__':
    run()
        


from collections import deque

frame_stack = deque(maxlen=4)

state, _ = env.reset()

# preprocess first frame 4 times
first_frame = agent.preprocess_frames([state])[0]   # single frame
for _ in range(4):
    frame_stack.append(first_frame)

while not done:
    # stack to tensor (4,84,84)
    stacked_state = np.stack(frame_stack, axis=0)

    # choose action
    action = agent.choose_action(stacked_state)

    # take action
    next_state, reward, done, truncated, info = env.step(action)

    # preprocess new frame
    processed = agent.preprocess_frames([next_state])[0]
    frame_stack.append(processed)

    # next stacked state
    stacked_next = np.stack(frame_stack, axis=0)

    # store transition
    agent.store_transition(stacked_state, action, reward, stacked_next, done)

    # learn
    agent.learn()

    state = next_state



        
        

        

        
        


