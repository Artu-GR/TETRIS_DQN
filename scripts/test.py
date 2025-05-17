import sys

import torch
import pygame
import numpy as np
from rl import dqn_agent, tetris_env

pygame.init()

def auto_play(gpx=None):
    # Load environment
    env = tetris_env.TetrisEnv()
    clock = pygame.time.Clock()

    # Initialize agent with same architecture
    agent = dqn_agent.DQN_Agent(state_size=env.get_state().shape[0], action_size=env.action_space.n)

    # Load the trained model weights
    checkpoint_path = "models/trained_model.pth"
    agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    agent.q_network.eval()

    # Set epsilon to 0 (pure exploitation)
    agent.epsilon = 0.0

    # Reset the game
    state = env.reset_game()
    done = False

    gpx.draw_training(env.engine)

    while not done:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.select_action(state)
        next_state, reward, done, info = env.step(ep=0, action=action, gpx=gpx)
        state = next_state
        print(env.high_level_actions[action])

        if done:
            #gpx.draw_GameOver(env.engine)
            gpx.wait_ending_screen(env.engine)
        else:
            env.engine.update()
            gpx.drawBoard(None, env.engine)
            pygame.display.flip()
