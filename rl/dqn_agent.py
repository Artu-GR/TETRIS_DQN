import os
import random
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

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

            if data is None:
                print(f"Warning: Got None from tree.get({s}) at segment {i}")
                continue  # Skip this one — or handle differently
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        if not batch:
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
    
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def get_all(self):
        return list(self.buffer)

class DQN_Agent():
    def __init__(self, state_size=223, action_size=6, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.q_diff = 0
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        #self.replay_buffer = ReplayBuffer(capacity=100000)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6) # 100k size buffer (for 100k games training)
        self.t_buffer = ReplayBuffer(capacity=1000)

        self.batch_size = 64
        self.gamma = 0.99

        #Epsilon decay handling
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.05
        self.epsilon_start = 1
        self.epsilon_decay_steps = 80000 # 80k

        self.update_target_every = 1000 # 1k
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

    def store_transition(self, state, action, reward, next_state, done):
        # build transition tuple
        #td_error = reward + (np.random.rand() * 0.7)
        transition = (state, action, reward, next_state, done)

        # n-step logic (unchanged)
        self.n_buffer.append(transition)
        if len(self.n_buffer) == self.n_step:
            R, ns, d = self._get_n_step_info()
            s0, a0, _, _, _ = self.n_buffer[0]
            t0 = (s0, a0, R, ns, d)

            with torch.no_grad():
                s0_tensor = torch.tensor(s0, dtype=torch.float32).unsqueeze(0).to(self.device)
                ns_tensor = torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(self.device)

                q_value = self.q_network(s0_tensor)[0, action].item()
                next_q_value = self.target_network(ns_tensor).max(1)[0].item() if not d else 0.0
                td_error = reward + self.gamma * next_q_value - q_value

            self.replay_buffer.push(t0, td_error)

        if done:
            while len(self.n_buffer) > 0:
                R, ns, d = self._get_n_step_info()
                s0, a0, _, _, _ = self.n_buffer[0]
                t0 = (s0, a0, R, ns, d)

                with torch.no_grad():
                    s0_tensor = torch.tensor(s0, dtype=torch.float32).unsqueeze(0).to(self.device)
                    ns_tensor = torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(self.device)

                    q_value = self.q_network(s0_tensor)[0, action].item()
                    next_q_value = self.target_network(ns_tensor).max(1)[0].item() if not d else 0.0
                    td_error = reward + self.gamma * next_q_value - q_value

                self.replay_buffer.push(t0, td_error)
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

    def update(self, episode, beta=0.4):        
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
            #raise ValueError('Not enough states')
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
            #target = rewards + self.gamma * next_q * (1 - dones)

        self.q_diff = torch.abs(q_values - target_q).mean()

        # 4) weighted MSE loss
        td_errors = (target_q - q_values).detach().cpu().squeeze().numpy()

        loss = (nn.SmoothL1Loss()(q_values, target_q))
        #loss = (weights * (q_values - target_q).pow(2)).mean()

        # 5) gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        #loss = torch.clamp(loss, min=-1.0, max=1.0)

        # 6) update priorities in buffer
        self.replay_buffer.update_priorities(idxs, td_errors)

        # 7) periodic target‐network sync
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()

        # 8) decay ε
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - ((episode) / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_min)
        )
        self.last_loss = loss.item()

class DQNTrainingLogger:
    def __init__(self, log_dir="logs7"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.rewards = []
        self.losses = []
        self.param_changes = []

    def log_reward(self, reward):
        self.rewards.append(reward)

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_param_change(self, q_net, target_net):
        q_params = np.concatenate([p.data.cpu().numpy().flatten() for p in q_net.parameters()])
        target_params = np.concatenate([p.data.cpu().numpy().flatten() for p in target_net.parameters()])
        change = np.mean(np.abs(q_params - target_params))
        self.param_changes.append(change)

    def save(self):
        rewards_py = [float(r) for r in self.rewards]
        losses_py = [float(l) for l in self.losses]
        param_changes_py = [float(p) for p in self.param_changes]

        with open(os.path.join(self.log_dir, "training_logs.json"), "w") as f:
            json.dump({
                "rewards": rewards_py,
                "losses": losses_py,
                "param_changes": param_changes_py
            }, f)

    def load(self):
        with open(os.path.join(self.log_dir, "training_logs.json"), "r") as f:
            data = json.load(f)
            self.rewards = data["rewards"]
            self.losses = data["losses"]
            self.param_changes = data["param_changes"]

    def plot(self, ep, save_only=True):
        if save_only:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        #import matplotlib.pyplot as plt

        episodes = range(len(self.rewards))

        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        axs[0].plot(episodes, self.rewards, label="Average Reward")
        axs[0].set_title("Reward per Episode")
        axs[0].set_ylabel("Reward")
        axs[0].grid()

        axs[1].plot(range(len(self.losses)), self.losses, label="Loss")
        axs[1].set_title("Loss over Time")
        axs[1].set_ylabel("Loss")
        axs[1].grid()

        axs[2].plot(range(len(self.param_changes)), self.param_changes, label="Parameter Change")
        axs[2].set_title("Target Network Parameter Change")
        axs[2].set_ylabel("|Q - Target| Mean")
        axs[2].grid()

        for ax in axs:
            ax.legend()

        axs[2].set_xlabel("Episodes")
        plt.tight_layout()

        if save_only:
            plt.savefig(os.path.join(self.log_dir, f"training_plot_ep{ep}.png"))
            plt.close(fig)  # Close to free memory and avoid Tkinter errors
        else:
            plt.show()