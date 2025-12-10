import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical




class ActorNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims, fc2_dims, device):
        super(ActorNet, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)

        return x
    

class CriticNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims, fc2_dims, device):
        super(CriticNet, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.device = device

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Agent:
    def __init__(self, env, epochs, gamma, lr, fc1_dims, fc2_dims, device, mem_size, batch_size):

        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs

        self.input_dims = np.array(env.observation_space.shape).prod()
        self.n_actions = env.action_space.n


        self.actor_net = ActorNet(self.lr, self.input_dims, self.n_actions, self.fc1_dims, self.fc2_dims, device)
        self.critic_net = CriticNet(self.lr, self.input_dims, self.n_actions, self.fc1_dims, self.fc2_dims, device)


    def choose_action(self, state):
        with t.no_grad():
            state = t.tensor([state]).to(self.device)
            probs = self.actor_net.forward(state)
            dist = Categorical(probs)
            action = dist.sample()

        return action.cpu().detach().numpy()[0]

    
    
    def learn(self, state, action, reward, next_state, done):
        
        state = t.tensor([state], dtype=t.float32).to(self.device)
        next_state = t.tensor([next_state], dtype=t.float32).to(self.device)
        reward = t.tensor(reward, dtype=t.float32).to(self.device)
        action = t.tensor(action, dtype=t.long).to(self.device)

        probs = self.actor_net.forward(state)
        values = self.critic_net.forward(state).squeeze()
        next_values = self.critic_net.forward(next_state).squeeze()

        target_returns = reward + (self.gamma * next_values * (1 - int(done))) 
        adv = (target_returns - values).detach()

        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        log_prob = dist.log_prob(action)

        actor_loss = -(log_prob * adv + 0.01 * entropy)
        critic_loss = F.mse_loss(values, target_returns.detach())

        self.actor_net.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_net.optimizer.step()

        self.critic_net.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net.optimizer.step()
        

        


def run():

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    env_name = 'LunarLander-v2'
    
    env = gym.make(env_name)
    
    batch_size = 8
    memory_size = 8
    epochs = 3
    lr=1e-4
    gamma = 0.99

    agent = Agent(env, epochs=epochs, gamma=gamma, lr=lr, 
                  fc1_dims=256, fc2_dims=256, device=device, mem_size=memory_size, batch_size=batch_size)
    
    total_steps = 1000000
    n_episodes = 2000
    step = 0
    score_history = []
    episode_score = 0
    episode = 0

    state, _ = env.reset()

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward

            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        print(f'episode : {episode}, score : {score}, avg_score : {avg_score}')


if __name__=="__main__":
    run()




        