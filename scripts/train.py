import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pygame
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from rl import dqn_agent, tetris_env
from game import Graphics
from services import utils

pygame.init()

def train_model(episodes=5000):
    env = tetris_env.TetrisEnv()
    gpx = Graphics.Graphics()
    logger = dqn_agent.DQNTrainingLogger()
    state_size = env.get_state().shape[0]
    action_size = env.action_space.n

    agent = dqn_agent.DQN_Agent(state_size, action_size) # (223, 6)

    clock = pygame.time.Clock()

    start_episode = 0
    ckpt_path     = "checkpoints/checkpoint.pth.tar"
    lines = 0
    
    # try:
    #     start_episode, lines = utils.load_checkpoint(
    #         ckpt_path,
    #         model     = agent.q_network,
    #         optimizer = agent.optimizer,
    #         memory    = agent.replay_buffer
    #     )
    # except Exception as e:
    #     print(f"Failed to load checkpoint '{ckpt_path}': {e}")
    #     start_episode = 0

    tot_lines = 0
    tot_lines += lines
    ep = 0

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
                    utils.save_checkpoint({
                        'epoch': ep,
                        'state_dict': agent.q_network.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'memory': agent.replay_buffer,
                        'lines_cleared': tot_lines
                    })
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(ep, action, gpx)
            # if done:
            #     gpx.draw_GameOver(env.engine)
            # else:
            #     env.engine.update()
            #     gpx.drawBoard(None, env.engine)
            #     pygame.display.flip()
            lines = info.get('lines_cleared', 0)

            agent.store_transition(state, action, reward, next_state, done)

            tot_lines += lines

            env.engine.update()
            gpx.drawBoard(None, env.engine)
            pygame.display.flip()

            try:
                agent.update(ep)
            except Exception as e:
                print(f"Update failed at episode {ep}: {e}")
                #Update failed at episode 884: 'NoneType' object is not iterable
            state = next_state
            total_reward += reward

        #     env.engine.update()
        # gpx.drawBoard(None, env.engine)
        # pygame.display.flip()

        print(f"Episode {ep} | Total Reward: {total_reward: .3f} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.last_loss:.3f} | Total lines cleared: {tot_lines:.3f} | Buffer size: {len(agent.replay_buffer):.3f}")

        logger.log_reward(total_reward)
        logger.log_loss(agent.last_loss)
        logger.log_param_change(agent.q_network, agent.target_network)
        #logger.log_q_diff(agent.q_diff.item())

        if ep % agent.update_target_every == 0: #every 1000 games save a record
        #if ep % 1000 == 0:
            torch.save(agent.q_network.state_dict(), f"models7/dqn_episode_{ep}.pth")
            logger.save()
            logger.plot(ep, save_only=True)


    torch.save(agent.q_network.state_dict(), f"models7/trained_model.pth")
    utils.save_checkpoint({
            'epoch': episodes,
            'state_dict': agent.q_network.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'memory': agent.replay_buffer,
            'lines_cleared': tot_lines
        })
    
    logger.plot(episodes, save_only=True)

if __name__ == '__main__': 
    train_model(episodes=100000) # 100k