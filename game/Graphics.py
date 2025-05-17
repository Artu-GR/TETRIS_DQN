import pygame
import random

from .TetrisEngine import GameState
from scripts import test

pygame.init()
pygame.mixer.init()
# MAKE NEXT LINE A COMMENT TO LISTEN TO MUSIC
# pygame.mixer.music.set_volume(0)
ClearSFX = pygame.mixer.Sound("assets/audio/sfx/ClearLine.mp3")
GameOverSFX = pygame.mixer.Sound("assets/audio/sfx/GameOver.mp3")
RotationSFX = pygame.mixer.Sound("assets/audio/sfx/Rotation.mp3")
TetrisSFX = pygame.mixer.Sound("assets/audio/sfx/TetrisClear.mp3")

class Graphics():
    def __init__(self):
        self.screen_w = 300
        self.screen_h = 600
        self.rows = 20
        self.cols = 10
        self.TILE_SIZE = self.screen_h // self.rows
        self.colors = {
            0: pygame.Color('black'),
            'I': pygame.Color(100, 149, 237), #light blue
            'O': pygame.Color('yellow'),
            'T': pygame.Color('purple'),
            'L': pygame.Color('orange'),
            'J': pygame.Color('blue'),
            'S': pygame.Color('green'),
            'Z': pygame.Color('red')
        }
        self.music_stopped = False

        # Side spaces
        # self.extra_height = 20 (TILE_SIZE)
        self.side_width = 260
        self.side_rows = 8
        self.side_cols = 4

    def drawTitleScreen(self):
        pygame.event.clear()
        pygame.mixer.music.load('assets/audio/TitleScreenMusic.mp3')
        pygame.mixer.music.play(-1)
        screen = pygame.display.set_mode((self.screen_w + self.side_width * 2,
                                          self.screen_h + self.TILE_SIZE * 2))
        screen.fill(pygame.Color("#929292"))

        font = pygame.font.Font("assets/fonts/BungeeTint-Regular.ttf", 120)
        text = font.render('Tetris', True, (255, 255, 255))  
        font = pygame.font.Font("assets/fonts/RubikMonoOne-Regular.ttf", 28)
        text2 = font.render('Start', True, (0, 0, 0))  
        text3 = font.render('Auto-play', True, (0,0,0))
        border_color = pygame.Color('black')
        
        # TITLE
        screen.blit(text, (self.side_width + self.screen_w // 2 - text.get_width() // 2, 
                        screen.get_height() // 2 - text.get_height() // 2 - 200))  # Center the text
        
        # BUTTON 1
        button1_rect = pygame.Rect(
            self.side_width + self.screen_w // 2 - 125,  # 125 = 250 // 2
            screen.get_height() // 2 - 30,              # 30 = 60 // 2
            250, 60
        )
        border1 = button1_rect.inflate(6,6)
        pygame.draw.rect(screen, border_color, border1)
        pygame.draw.rect(screen, pygame.Color('#c1c0c0'), button1_rect)

        text2_pos = (
            button1_rect.x + (button1_rect.width - text2.get_width()) // 2,
            button1_rect.y + (button1_rect.height - text2.get_height()) // 2
        )
        screen.blit(text2, text2_pos)
        
        # BUTTON 2
        button2_rect = pygame.Rect(
            self.side_width + self.screen_w // 2 - 120,  # 120 = 240 // 2
            screen.get_height() // 2 - 30 + 200,
            250, 60
        )
        border2 = button2_rect.inflate(6,6)
        pygame.draw.rect(screen, border_color, border2)
        pygame.draw.rect(screen, pygame.Color('#c1c0c0'), button2_rect)

        text3_pos = (
            button2_rect.x + (button2_rect.width - text3.get_width()) // 2,
            button2_rect.y + (button2_rect.height - text3.get_height()) // 2
        )
        screen.blit(text3, text3_pos)


        waiting = True
        colors = [
            pygame.Color("cyan"), pygame.Color("blue"), pygame.Color("orange"),
            pygame.Color("yellow"), pygame.Color("green"), pygame.Color("purple"),
            pygame.Color("red")
            ]
        
        last_color_change = pygame.time.get_ticks()  # Save initial time 
        color_change_interval = 300
        current_color=random.choice(colors)
        while waiting:
            screen_width, screen_height = screen.get_size()
            TILE_SIZE = 30
            
            current_time = pygame.time.get_ticks()
            if current_time - last_color_change >= color_change_interval:
                last_color_change = current_time  # Update time of last change
                current_color = random.choice(colors)  # Change random color 

            #borders 
            for x in range(0, screen_width + TILE_SIZE, TILE_SIZE):
                for y in [0, screen_height - TILE_SIZE]:
                    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(screen,current_color, rect)  
                    pygame.draw.rect(screen, pygame.Color("black"), rect, 1) 
            
            for y in range(0, screen_height + TILE_SIZE, TILE_SIZE):
                for x in [0, screen_width - TILE_SIZE]:
                    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(screen, current_color, rect)  
                    pygame.draw.rect(screen, pygame.Color("black"), rect, 1) 
            
            pygame.display.flip()  # Update the display

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    col = location[0]
                    row = location[1]

                    if (305 <= row <= 365) and (281 <= col <= 521):
                        #print("Startgame selected")
                        waiting = False  # Salir del bucle cuando se presiona una tecla / Continuar con el juego
                        pygame.mixer.music.stop()
                        self.main()
                    elif (505 <= row <= 565) and (281 <= col <= 521):
                        # AUTO-PLAY (AI)
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load('assets/audio/TetrisMusic.mp3')
                        pygame.mixer.music.play(-1)
                        test.auto_play(self)
                        #print("Autoplay selected")
                        pass
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False  # Salir del bucle cuando se presiona una tecla
                        pygame.mixer.music.stop()
                        self.main()

    def main(self, mode=None):
        pygame.mixer.music.load('assets/audio/TetrisMusic.mp3')
        screen = pygame.display.get_surface()
        if screen == None:
            screen = pygame.display.set_mode((self.screen_w + self.side_width * 2,
                                          self.screen_h + self.TILE_SIZE * 2))
        clock = pygame.time.Clock()
        screen.fill(pygame.Color("grey73"))
        gs = GameState()
        self.drawBorder(screen)
        self.draw_sides(screen, gs)
        pygame.display.update()
        pygame.mixer.music.play(-1)
        running = True
        if mode == 'AI':
            self.AI(screen, gs, clock)
        while(running):
            clock.tick(60)
            for action in pygame.event.get():
                if action.type == pygame.QUIT or gs.game_ended:
                    running = False
                elif action.type == pygame.KEYDOWN:
                    if not self.key_handling(action.key, gs):
                        break
            if gs.is_paused:
                self.show_pause_screen()
            elif gs.game_ended:
                break
            else:
                gs.update()
                self.drawBoard(screen, gs)
                pygame.display.flip()
        
        if gs.game_ended:
            self.wait_ending_screen(gs)

    def key_handling(self, action, gs):
        screen = pygame.display.get_surface()

        if (gs.is_paused and action != pygame.K_ESCAPE):
            return # continue
        elif gs.game_ended:
            return False # break
        
        elif (gs.is_paused and action == pygame.K_ESCAPE):
            screen.fill(pygame.Color('grey73'))
            self.drawBorder(screen)
            self.draw_sides(screen, gs)
            self.play_music()
            self.music_stopped = False

        if action == pygame.K_LEFT:
            gs.moveLeft()
        elif action == pygame.K_RIGHT:
            gs.moveRight()
        elif action == pygame.K_UP:
            gs.rotatePiece()
        elif action == pygame.K_DOWN:
            gs.moveDown()
        elif action == pygame.K_c:
            gs.hold_Piece()
        elif action == pygame.K_SPACE:
            gs.dropPiece()

        elif action == pygame.K_ESCAPE:
            gs.is_paused = not gs.is_paused
            if gs.is_paused:
                self.stop_music()  # Pause music if game is paused
                self.music_paused = True  # Track that music is paused
            else:
                self.play_music()  # Unpause music when game is unpaused
                self.music_paused = False

    def wait_ending_screen(self, gs):
        self.draw_GameOver(gs)
            
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    col = location[0]
                    row = location[1]

                    if (390 <= row <= 450) and (280 <= col <= 520): #(280 <= col <= 520)
                        del(gs)
                        waiting = False  # Salir del bucle cuando se presiona una tecla
                        pygame.mixer.music.stop()
                        self.drawTitleScreen()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        del(gs)
                        waiting = False  # Salir del bucle cuando se presiona una tecla
                        pygame.mixer.music.stop()
                        pygame.event.clear()
                        self.drawTitleScreen()

    def play_music(self):
        pygame.mixer.music.unpause()

    def stop_music(self):
        pygame.mixer.music.pause()

    def show_pause_screen(self):
        font = pygame.font.Font("assets/fonts/BungeeShade-Regular.ttf", 72)
        text = font.render('Paused', True, (255, 255, 255))  # White text
        screen = pygame.display.get_surface()
        screen.fill((0, 0, 0))  # Fill the screen with black
        screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2, 
                        screen.get_height() // 2 - text.get_height() // 2))  # Center the text
        pygame.display.flip()  # Update the display

    def drawBoard(self, screen, gs):
        screen = pygame.display.get_surface()

        for row in range(self.rows):
            for col in range(self.cols):
                pygame.draw.rect(screen, self.colors[gs.board[row][col]],
                                 pygame.Rect(col * self.TILE_SIZE + self.side_width, row * self.TILE_SIZE + self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
        self.draw_sides(screen, gs)
        self.drawProyection(screen, gs)

    def drawProyection(self, screen, gs):
        for r,c in gs.projected_coords:
            pygame.draw.rect(screen, self.colors[gs.currentPiece.type],
                    pygame.Rect(c * self.TILE_SIZE + self.side_width,
                               r * self.TILE_SIZE + self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
            pygame.draw.rect(screen, pygame.Color('black'),
                    pygame.Rect(c * self.TILE_SIZE + self.side_width + 1,
                               r * self.TILE_SIZE + self.TILE_SIZE + 1, self.TILE_SIZE - 2, self.TILE_SIZE - 2))

    def draw_sides(self, screen, gs):
        # left side (hold piece and score)
        for row in range(4):
            for col in range(4):
                pygame.draw.rect(screen, self.colors[gs.holdPieceGrid[row-1][col]] if 1<= row<= 2 else pygame.Color('black'),
                    pygame.Rect(col * self.TILE_SIZE + (self.side_width - self.side_cols * self.TILE_SIZE) // 2 - 10,
                                row * self.TILE_SIZE + self.TILE_SIZE + (self.screen_h // 2 - self.side_rows * self.TILE_SIZE),
                                self.TILE_SIZE, self.TILE_SIZE))
                
        pygame.draw.rect(screen, pygame.Color('grey48'),
                    pygame.Rect(self.side_width // 2 - 75,
                                (self.TILE_SIZE*2 + (self.screen_h // 2) + 35),
                                self.TILE_SIZE * 4 + 10, self.TILE_SIZE * 3))
        
        #font = pygame.font.Font("assets/fonts/RubikMonoOne-Regular.ttf", 40)
        font = pygame.font.SysFont("Arial", 48)
        text = font.render(str(gs.score), True, (0,0,0))  # White text
        screen.blit(text, (self.side_width // 2 - 75 + (self.TILE_SIZE * 4 + 10 - text.get_width()) // 2, 
                        (self.TILE_SIZE*2 + (self.screen_h // 2) + 50)))

        # right side (next pieces)
        for row in range(self.side_rows):
            for col in range(self.side_cols):
                pygame.draw.rect(screen, self.colors[gs.nextPiecesGrid[row][col]],
                    pygame.Rect(col * self.TILE_SIZE + self.screen_w + self.side_width + (self.side_width - self.side_cols * self.TILE_SIZE) // 2,
                                row * self.TILE_SIZE + self.TILE_SIZE + (self.screen_h // 2 - self.side_rows * self.TILE_SIZE),
                                self.TILE_SIZE, self.TILE_SIZE))            
    
    def drawBorder(self, screen):
        font = pygame.font.Font("assets/fonts/RubikMonoOne-Regular.ttf", 40)
        text = font.render('Hold', True, (255, 255, 255))  # White text
        screen.blit(text, (self.side_width // 2 - text.get_width() // 2 - 10, 
                        (self.TILE_SIZE*2 + (self.screen_h // 2 - self.side_rows * self.TILE_SIZE)) // 2 - text.get_height() // 2))
        text2 = font.render('Next', True, (255, 255, 255))  # White text
        screen.blit(text2, (self.side_width // 2 - text.get_width() // 2 + self.side_width + self.screen_w, 
                        (self.TILE_SIZE*2 + (self.screen_h // 2 - self.side_rows * self.TILE_SIZE)) // 2 - text.get_height() // 2))
        text3 = font.render('Score', True, (255, 255, 255))  # White text
        screen.blit(text3, (self.side_width // 2 - text.get_width() // 2 - 30, 
                        (self.TILE_SIZE*2 + (self.screen_h // 2)) - text.get_height() // 2))

        for row in range(self.rows+2):
            for col in range(self.cols+2):
                pygame.draw.rect(screen, pygame.Color('black'),
                                 pygame.Rect(col * self.TILE_SIZE + self.side_width - self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

                pygame.draw.rect(screen, pygame.Color('grey48'),
                                 pygame.Rect(col * self.TILE_SIZE + self.side_width - self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE-2, self.TILE_SIZE-2))

    def draw_GameOver(self, gs):
        game_over_w = self.screen_w
        game_over_h = (self.screen_w) * 9 // 16

        self.stop_music()
        self.music_paused = True
        GameOverSFX.play()

        font = pygame.font.Font("assets/fonts/BungeeShade-Regular.ttf", 70)
        font2 = pygame.font.Font("assets/fonts/RubikMonoOne-Regular.ttf", 28)
        text = font.render('Game Over', True, (255, 255, 255))  # White text
        text2 = font.render(f'Score: {gs.score}', True, (255, 255, 255))  # White text
        text3 = font2.render('Play again', True, (0,0,0))
        screen = pygame.display.get_surface()

        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)  # Transparent surface
        overlay.fill((0, 0, 0, 192))  # Semi-transparent black (RGBA: 192 = 25% transparency)

        # Fill the screen with the translucent overlay
        screen.blit(overlay, (0, 0))
        
        screen.blit(text, (self.side_width + game_over_w // 2 - text.get_width() // 2, 
                        screen.get_height() // 2 - text.get_height() // 2 - game_over_h // 2))  # Center the text
        screen.blit(text2, (self.side_width + game_over_w // 2 - text2.get_width() // 2, 
                        screen.get_height() // 2 - text2.get_height() // 2))  # Center the text
        
        button3_rect = pygame.Rect(self.side_width + game_over_w // 2 - 250 // 2,
        screen.get_height() // 2 - 60 // 2 + game_over_h // 2, 250,60)

        border3 = button3_rect.inflate(6,6)
        pygame.draw.rect(screen,"black", border3)
        pygame.draw.rect(screen, "grey73", button3_rect)

        # Centrar el texto dentro del botón
        text3_pos = (
            button3_rect.x + (button3_rect.width - text3.get_width()) // 2,
            button3_rect.y + (button3_rect.height - text3.get_height()) // 2
        )
        screen.blit(text3, text3_pos)

        pygame.display.flip()  # Update the display

    def draw_training(self, gs):
        screen = pygame.display.set_mode((self.screen_w + self.side_width * 2,
                                          self.screen_h + self.TILE_SIZE * 2))
        screen.fill(pygame.Color('grey73'))
        self.draw_sides(screen, gs)
        self.drawBorder(screen)
        self.drawBoard(screen, gs)
        self.play_music()