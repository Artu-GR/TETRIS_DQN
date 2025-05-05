
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    
# class ReplayBuffer():
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def size(self):
#         return len(self.buffer)

import numpy as np
import random

class SumTree:
    """Binary tree where parent’s value = sum of children. Leaf nodes store priorities."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = [None]*capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1)//2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps   = 1e-6  # small amount to avoid zero priority

    # def push(self, transition, td_error=None):
    #     # new transitions get max priority so they’re likely sampled at least once
    #     priority = (abs(td_error) + self.eps)**self.alpha if td_error is not None else self.tree.tree.max() or 1.0
    #     self.tree.add(priority, transition)
    def push(self, transition, td_error=None):
        max_priority = self.tree.tree.max()
        if max_priority == 0:
            max_priority = 1.0
        priority = (abs(td_error) + self.eps)**self.alpha if td_error is not None else max_priority
        self.tree.add(priority, transition)


    def sample(self, batch_size, beta=0.4):
        if self.tree.total() == 0:
            print("SumTree total priority is zero!")
            return [], [], []
        
        batch = []
        idxs  = []
        seg   = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.uniform(i*seg, (i+1)*seg)
            idx, p, data = self.tree.get(s)

            # if data is None:
            #         print(f"Warning: Got None from tree.get({s}) at segment {i}")
            #         continue  # Skip this one — or handle differently
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        if not batch:
            print("EMPTY BATCH 2")
            return [], [], []
        #sampling_prob = np.array(priorities) / self.tree.total()
        sampling_prob = np.array(priorities) / self.tree.total()
        #is_weight = (self.tree.size * sampling_prob) ** (-beta)
        is_weight = (self.tree.size * sampling_prob + self.eps) ** (-beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.eps)**self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size

class DQN_Agent():
    def __init__(self, state_size=223, action_size=6, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        #self.replay_buffer = ReplayBuffer(capacity=100000)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=75000, alpha=0.6)

        self.batch_size = 64
        self.gamma = 0.99

        #Epsilon decay handling
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.epsilon_start = 1
        self.epsilon_decay_steps = 4000 # 4500

        self.update_target_every = 100
        self.step_count = 0
        self.last_loss = 0

        self.n_step = 3
        self.n_buffer = deque(maxlen=self.n_step)

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state)

        return q_values.argmax().item()

    # def store_transition(self, state, action, reward, next_state, done):
    #     self.replay_buffer.push(state, action, reward, next_state, done)
    def store_transition(self, state, action, reward, next_state, done):
        # push into n-step buffer
        self.n_buffer.append((state, action, reward, next_state, done))
        #print("Replay Buffer Size:", len(self.replay_buffer))
        if len(self.n_buffer) == self.n_step:
            r, ns, d = self._get_n_step_info()
            s0, a0, _, _, _ = self.n_buffer[0]
            self.replay_buffer.push((s0, a0, r, ns, d))

        if done:
            # flush remaining
            while len(self.n_buffer) > 0:
                r, ns, d = self._get_n_step_info()
                s0, a0, _, _, _ = self.n_buffer[0]
                self.replay_buffer.push((s0, a0, r, ns, d))
                self.n_buffer.popleft()

    def _get_n_step_info(self):
        """Compute n-step return from the buffer."""
        R, next_state, done = 0, None, False
        for idx, (_, _, reward, ns, d) in enumerate(self.n_buffer):
            R += (self.gamma ** idx) * reward
            next_state = ns
            done = d
            if d:
                break
        return R, next_state, done

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # def update(self, episode):
    #     if self.replay_buffer.size() < 5000:
    #         return
        
    #     #batch = self.replay_buffer.sample(self.batch_size)
    #     batch, idxs, is_weights = self.replay_buffer.sample(self.batch_size, beta=beta)
    #     states, actions, rewards, next_states, dones = zip(*batch)

    #     states = torch.FloatTensor(np.array(states)).to(self.device) # Shape: (batch_size, 224)
    #     actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
    #     rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
    #     next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
    #     dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

    #     q_values = self.q_network(states).gather(1, actions)

    #     # with torch.no_grad():
    #     #     next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
    #     #     target_q = rewards + self.gamma * next_q_values * (1 - dones)

    #     with torch.no_grad():
    #         next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
    #         target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

    #     loss = nn.MSELoss()(q_values, target_q)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     self.step_count += 1
    #     if self.step_count % self.update_target_every == 0:
    #         self.update_target_network()

    #     #Epsilon decay
    #     #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    #     self.epsilon = max(
    #         self.epsilon_min,
    #         self.epsilon_start - (episode / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_min)
    #     )

    #     self.last_loss = loss.item()

    def update(self, episode, beta=0.4):
        # don’t start until we have plenty of experience
        # if len(self.replay_buffer) < 1000:
        #     return
        
        # 1) don’t start unless we can sample a full batch
        if len(self.replay_buffer) < self.batch_size:
            return
 
        # 2) avoid division by zero when priorities sum to zero
        total_p = self.replay_buffer.tree.total()
        if total_p <= 0:
            return
 
        # 3) if sampling somehow fails or returns too few, bail out
        sample = self.replay_buffer.sample(self.batch_size, beta=beta)
        if (sample is None) or (len(sample[0]) < self.batch_size):
            return
        batch, idxs, is_weights = sample

        # 1) sample with priorities + importance weights
        #batch, idxs, is_weights = self.replay_buffer.sample(self.batch_size, beta=beta)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights     = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # 2) current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # 3) n-step bootstrap target
        with torch.no_grad():
            next_q   = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        # 4) weighted MSE loss
        td_errors = (target_q - q_values).detach().cpu().squeeze().numpy()
        loss = (weights * (q_values - target_q).pow(2)).mean()

        # 5) gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) update priorities in buffer
        self.replay_buffer.update_priorities(idxs, td_errors)

        # 7) periodic target‐network sync
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()

        # 8) decay ε
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - (episode / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_min)
        )
        self.last_loss = loss.item()
