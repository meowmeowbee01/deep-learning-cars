# https://www.pygame.org/docs/#pygame-front-page

import sys
import math
import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Deep Learning Cars"

ACCELERATION = 5
STEERING_SPEED = 5

MAX_SPEED = 5

PLAYER_SIZE = 10

# pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_speed = 0
player_direction = 0  # radians
player_change_angle = 0

while True:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    pygame.draw.circle(screen, "red", player_pos, PLAYER_SIZE)

    #inputs
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and not keys[pygame.K_s]:
        #accelerate forward
        player_speed += ACCELERATION * dt

    elif keys[pygame.K_s] and not keys[pygame.K_w]:
        #accelerate backwards
        player_speed -= ACCELERATION * dt

    if keys[pygame.K_a] and not keys[pygame.K_d]:
        #turn right
        player_change_angle = STEERING_SPEED * dt

    elif keys[pygame.K_d] and not keys[pygame.K_a]:
        #turn left
        player_change_angle = -STEERING_SPEED * dt

    elif not keys[pygame.K_d] and not keys[pygame.K_a]:
        #dont turn
        player_change_angle = 0

    elif keys[pygame.K_d] and keys[pygame.K_a]:
        #dont turn
        player_change_angle = 0

    #update direction
    player_direction += player_change_angle

    #cap speed
    player_speed = min(MAX_SPEED, player_speed)
    player_speed = max(-MAX_SPEED, player_speed)

    #calculate new postion
    player_pos.x += player_speed * math.sin(player_direction)
    player_pos.y += player_speed * math.cos(player_direction)

    # oob
    if player_pos.x < 0 + PLAYER_SIZE:
        player_pos.x = 0 + PLAYER_SIZE
    if player_pos.x > screen.get_width() - PLAYER_SIZE:
        player_pos.x = screen.get_width() - PLAYER_SIZE
    if player_pos.y < 0 + PLAYER_SIZE:
        player_pos.y = 0 + PLAYER_SIZE
    if player_pos.y > screen.get_height() - PLAYER_SIZE:
        player_pos.y = screen.get_height() - PLAYER_SIZE

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000
