import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from rl import dqn_agent, tetris_env
from game import Graphics

import time
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

pygame.init()

def train_model(episodes=5000):
    env = tetris_env.TetrisEnv()
    gpx = Graphics.Graphics()
    state_size = env.get_state().shape[0]
    action_size = env.action_space.n
    # print(action_size)
    writer = SummaryWriter(log_dir="runs/tetris_dqn")

    # print(state_size, action_size)

    agent = dqn_agent.DQN_Agent(state_size, action_size) # (223, 6)

    clock = pygame.time.Clock()

    for ep in range(episodes):
        state = env.reset_game()
        gpx.draw_training(env.engine)
        pygame.display.flip()
        total_reward = 0
        done = False

        while not done:
            if ep >= 4000: clock.tick(3)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # gpx.drawBoard(None, env.engine)
            # pygame.display.flip()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # if done:
            #     gpx.draw_GameOver(env.engine)
            # else:
            #     env.engine.update()
            #     gpx.drawBoard(None, env.engine)
            #     pygame.display.flip()
            agent.store_transition(state, action, reward, next_state, done)
            env.engine.update()
            gpx.drawBoard(None, env.engine)
            pygame.display.flip()
            
            # agent.update(ep)
            try:
                agent.update(ep)
            except Exception as e:
                print(f"Update failed at episode {ep}: {e}")
            state = next_state
            total_reward += reward

        #     env.engine.update()
        # gpx.drawBoard(None, env.engine)
        # pygame.display.flip()

        print(f"Episode {ep} | Total Reward: {total_reward: .3f} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.last_loss:.3f} | Buffer size: {agent.replay_buffer.__len__():.3f}")
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Epsilon", agent.epsilon, ep)
        writer.add_scalar("Loss", agent.last_loss, ep)

        if ep % agent.update_target_every == 0:
            torch.save(agent.q_network.state_dict(), f"models/dqn_episode_{ep}.pth")

    writer.close()

if __name__ == '__main__':
    train_model(episodes=5000)