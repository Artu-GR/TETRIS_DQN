import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

from rl import dqn_agent, tetris_env
from game import Graphics
from services import utils

pygame.init()

def train_model(episodes=5000):
    env = tetris_env.TetrisEnv()
    gpx = Graphics.Graphics()
    state_size = env.get_state().shape[0]
    action_size = env.action_space.n
    # print(action_size)
    writer = SummaryWriter(log_dir="runs4/tetris_dqn")

    # print(state_size, action_size)

    agent = dqn_agent.DQN_Agent(state_size, action_size) # (223, 6)

    clock = pygame.time.Clock()

    # start_episode = 0
    # best_score = -float('inf')
    ckpt_path     = "checkpoints/checkpoint.pth.tar"

    try:
        start_episode, best_score = utils.load_checkpoint(
            ckpt_path,
            model     = agent.q_network,
            optimizer = agent.optimizer,
            memory    = agent.replay_buffer
        )
        print(f"Resumed from ep {start_episode}, best_score={best_score:.3f}")
    except Exception as e:
        print(f"Failed to load checkpoint '{ckpt_path}': {e}")
        start_episode = 0
        best_score    = -float('inf')

    for ep in range(start_episode, episodes):
        state = env.reset_game()
        gpx.draw_training(env.engine)
        pygame.display.flip()
        total_reward = 0
        done = False

        while not done:
            #clock.tick(60)
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

        print(f"Episode {ep} | Total Reward: {total_reward: .3f} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.last_loss:.3f} | Buffer size: {len(agent.replay_buffer):.3f}")
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Epsilon", agent.epsilon, ep)
        writer.add_scalar("Loss", agent.last_loss, ep)

        is_best = total_reward > best_score
        if is_best:
            best_score = total_reward
            print("UPDATED MODEL")

        utils.save_checkpoint({
            'epoch': ep,
            'state_dict': agent.q_network.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'memory': agent.replay_buffer,
            'best_score': best_score,
        }, is_best)

        # if ep % agent.update_target_every == 0:
        #     torch.save(agent.q_network.state_dict(), f"models4/dqn_episode_{ep}.pth")

    writer.close()
    torch.save(agent.q_network.state_dict(), f"checkpoints/trained_model.pth")

if __name__ == '__main__':
    train_model(episodes=5000)