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

    writer = SummaryWriter(log_dir="runs6/tetris_dqn")

    agent = dqn_agent.DQN_Agent(state_size, action_size) # (223, 6)

    clock = pygame.time.Clock()

    start_episode = 0
    ckpt_path     = "checkpoint.pth.tar"
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

            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            # q_values = agent.q_network(state_tensor)
            #print(f"Q-values for current state: {q_values.detach().cpu().numpy()}")

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action, gpx)
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
        
        writer.add_scalar("Reward/Total", total_reward, ep)
        writer.add_scalar("Epsilon", agent.epsilon, ep)
        writer.add_scalar("Loss", agent.last_loss, ep)

        if ep % agent.update_target_every == 0: #every 1000 games save a record
            torch.save(agent.q_network.state_dict(), f"models6/dqn_episode_{ep}.pth")


    writer.close()
    torch.save(agent.q_network.state_dict(), f"models6/trained_model.pth")
    utils.save_checkpoint({
            'epoch': episodes,
            'state_dict': agent.q_network.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'memory': agent.replay_buffer,
            'lines_cleared': tot_lines
        })

# def prefill_model():
#     PREFILL_THRESHOLD = 5000

#     env = tetris_env.TetrisEnv()
#     gpx = Graphics.Graphics()
#     state_size = env.get_state().shape[0]
#     action_size = env.action_space.n

#     writer = SummaryWriter(log_dir="runs5/tetris_dqn")

#     #print(state_size, action_size)

#     agent = dqn_agent.DQN_Agent(state_size, action_size) # (205, 41)

#     clock = pygame.time.Clock()

#     start_episode = 0
#     ckpt_path     = "checkpoints/checkpoint.pth.tar"
#     lines = 0
    
#     try:
#         start_episode, lines = utils.load_checkpoint(
#             ckpt_path,
#             model     = agent.q_network,
#             optimizer = agent.optimizer,
#             memory    = agent.replay_buffer
#         )
#     except Exception as e:
#         print(f"Failed to load checkpoint '{ckpt_path}': {e}")
#         start_episode = 0

#     tot_lines = 0
#     tot_lines += lines

#     ep = start_episode
#     while True:
#         lines_obj_reached = False
#         state = env.reset_game()
#         gpx.draw_training(env.engine)
#         pygame.display.flip()
#         total_reward = 0
#         done = False

#         while not done:
#             clock.tick(20)
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     sys.exit()

#             action = agent.select_action(state)
#             next_state, reward, done, info = env.step(action, gpx) 
#             # if done:
#             #     gpx.draw_GameOver(env.engine)
#             # else:
#             #     env.engine.update()
#             #     gpx.drawBoard(None, env.engine)
#             #     pygame.display.flip()
#             lines = info.get('lines_cleared', 0)

#             if len(agent.replay_buffer) <  PREFILL_THRESHOLD:
#                 if lines > 0:
#                     agent.store_transition(state, action, reward, next_state, done)
#             else:
#                 agent.store_transition(state, action, reward, next_state, done)
#             #agent.store_transition(state, action, reward, next_state, done)

#             tot_lines += lines
#             if (len(agent.replay_buffer) >= PREFILL_THRESHOLD):
#                 lines_obj_reached = True
#                 break

#             env.engine.update()
#             gpx.drawBoard(None, env.engine)
#             pygame.display.flip()

#             state = next_state
#             total_reward += reward

#         #     env.engine.update()
#         # gpx.drawBoard(None, env.engine)
#         # pygame.display.flip()

#         print(f"Game: {ep} | Total lines cleared: {tot_lines:.3f} | Buffer size: {len(agent.replay_buffer):.3f}")

#         if len(agent.replay_buffer) < PREFILL_THRESHOLD:
#             utils.save_checkpoint({
#                 'epoch': len(agent.replay_buffer),
#                 'state_dict': agent.q_network.state_dict(),
#                 'optimizer': agent.optimizer.state_dict(),
#                 'memory': agent.replay_buffer,
#                 'lines_cleared': tot_lines
#             })

#         if lines_obj_reached:
#             break
        
#         ep += 1

#     writer.close()
#     torch.save(agent.q_network.state_dict(), f"models5/trained_model.pth")

if __name__ == '__main__': 
    #prefill_model()
    train_model(episodes=100000)