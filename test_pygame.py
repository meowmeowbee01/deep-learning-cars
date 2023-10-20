# https://www.pygame.org/docs/#pygame-front-page

import os
import sys
import math
import pygame

SPRITE_SCALING = 0.05

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Deep Learning Cars"

ACCELERATION = 5
STEERING_SPEED = 5

MAX_SPEED = 8

# pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()
dt = 0

player_image = pygame.image.load("tank_real.png").convert_alpha()
player_image = pygame.transform.scale(
    player_image,
    (
        int(player_image.get_width() * SPRITE_SCALING),
        int(player_image.get_height() * SPRITE_SCALING),
    ),
)

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_speed = 0
player_direction = 0  # radians
player_change_angle = 0

# Calculate the radius of the player
player_radius = math.sqrt(
    (player_image.get_width() / 2) ** 2 + (player_image.get_height() / 2) ** 2
)

player_rect = player_image.get_rect(center=player_pos)

# Wall setup
walls = [
    pygame.Rect(100, 100, 600, 20),
    pygame.Rect(100, 100, 20, 400),
    pygame.Rect(100, 500, 600, 20),
    pygame.Rect(700, 100, 20, 400),
]


def cast_ray(angle):
    x, y = player_pos
    step_size = 5

    while 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)

        for wall in walls:
            if wall.collidepoint(x, y):
                return x, y

    return x, y


while True:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    rotated_player = pygame.transform.rotate(
        player_image, math.degrees(player_direction)
    )
    player_rect = rotated_player.get_rect(center=player_pos)
    screen.blit(rotated_player, player_rect.topleft)

    # inputs
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and not keys[pygame.K_s]:
        # accelerate forward
        player_speed += ACCELERATION * dt

    elif keys[pygame.K_s] and not keys[pygame.K_w]:
        # accelerate backwards
        player_speed -= ACCELERATION * dt

    if keys[pygame.K_a] and not keys[pygame.K_d]:
        # turn right
        player_change_angle = STEERING_SPEED * dt

    elif keys[pygame.K_d] and not keys[pygame.K_a]:
        # turn left
        player_change_angle = -STEERING_SPEED * dt

    elif not keys[pygame.K_d] and not keys[pygame.K_a]:
        # dont turn
        player_change_angle = 0

    elif keys[pygame.K_d] and keys[pygame.K_a]:
        # dont turn
        player_change_angle = 0

    # update direction
    player_direction += player_change_angle

    # cap speed
    player_speed = min(MAX_SPEED, player_speed)
    player_speed = max(-MAX_SPEED, player_speed)

    # calculate new postion
    player_pos.x += player_speed * math.sin(player_direction)
    player_pos.y += player_speed * math.cos(player_direction)

    # oob
    if player_pos.x < 0 + player_radius:
        player_pos.x = 0 + player_radius
    if player_pos.x > screen.get_width() - player_radius:
        player_pos.x = screen.get_width() - player_radius
    if player_pos.y < 0 + player_radius:
        player_pos.y = 0 + player_radius
    if player_pos.y > screen.get_height() - player_radius:
        player_pos.y = screen.get_height() - player_radius

    # walls
    for wall in walls:
        pygame.draw.rect(screen, (0, 0, 0), wall)

    for angle in range(0, 181, 30):
        ray_angle = math.radians(angle) - player_direction
        end_point = cast_ray(ray_angle)
        pygame.draw.line(
            screen, (255, 0, 0), player_pos, (int(end_point[0]), int(end_point[1]))
        )

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000
