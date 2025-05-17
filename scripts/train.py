import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pygame
import torch

from rl import dqn_agent, tetris_env

pygame.init()

def train_model(episodes=5000):
    env = tetris_env.TetrisEnv()
    logger = dqn_agent.DQNTrainingLogger()
    state_size = env.get_state().shape[0]
    action_size = env.action_space.n

    print(state_size, " ", action_size)

    agent = dqn_agent.DQN_Agent(state_size, action_size) # (235, 41)
    agent.epsilon_decay_steps = 0.85 * episodes

    start_episode = 0
    episode = 100000
    ckpt_path     = f"chkpts/checkpoint_ep{episode}.pt"
    lines = 0
    
    start_episode = agent.load_checkpoint(ckpt_path)

    tot_lines = 0
    tot_lines += lines
    ep = 0

    agent.epsilon_decay_steps = 0.85 * (episodes - start_episode)

    for ep in range(start_episode+1, episodes+1):
        state = env.reset_game()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(ep, action)

            lines = info.get('lines_cleared', 0)
            tot_lines += lines

            agent.store_transition(state, action, reward, next_state, done)

            if lines > 0:
                for _ in range(15): # 7 is not enough
                    agent.store_transition(state, action, reward, next_state, done)

            try:
                agent.update(ep-start_episode)
            except Exception as e:
                print(f"Update failed at episode {ep}: {e}")
            state = next_state
            total_reward += reward

        print(f"Episode {ep} | Total Reward: {total_reward: .3f} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.last_loss:.3f} | Total lines cleared: {tot_lines:.3f} | Buffer size: {len(agent.replay_buffer):.3f}")

        logger.log_reward(total_reward)
        logger.log_loss(agent.last_loss)
        logger.log_param_change(agent.q_network, agent.target_network)

        if ep % agent.update_target_every == 0: #every 1000 games save a record
            #torch.save(agent.q_network.state_dict(), f"models8/dqn_episode_{ep}.pth")
            agent.save_checkpoint(ep)
            logger.save()
            logger.plot(ep, save_only=True)


    torch.save(agent.q_network.state_dict(), f"models/trained_model.pth")
    
    logger.plot(episodes, save_only=True)

if __name__ == '__main__': 
    train_model(episodes=150000) # 100k base training