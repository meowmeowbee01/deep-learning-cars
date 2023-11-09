import os
import sys
import math
import pygame
import random

# Define your AI agent class here
class AI_Agent:
    def __init__(self):
        self.score = 0

    def play_game(self, game_state):
        # Implement your AI logic based on the game state
        # Example: Always move forward and occasionally steer left or right
        ai_actions = {
            "forward": True,
            "steer_left": random.random() < 0.1,  # 10% chance to steer left
            "steer_right": random.random() < 0.1,  # 10% chance to steer right
        }
        return ai_actions

SPRITE_SCALING = 0.05
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Deep Learning Cars"
ACCELERATION = 5
STEERING_SPEED = 5
MAX_SPEED = 8

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()
dt = 0

player_image = pygame.image.load("tank_real_2.png").convert_alpha()
player_image = pygame.transform.scale(
    player_image,
    (
        int(player_image.get_width() * SPRITE_SCALING),
        int(player_image.get_height() * SPRITE_SCALING),
    ),
)

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_speed = 0
player_direction = 0
player_change_angle = 0

player_radius = math.sqrt(
    (player_image.get_width() / 2) ** 2 + (player_image.get_height() / 2) ** 2
)

player_rect = player_image.get_rect(center=player_pos)

walls = [
    pygame.Rect(100, 100, 600, 20),
    pygame.Rect(100, 100, 20, 400),
    pygame.Rect(100, 500, 200, 20),
    pygame.Rect(700, 100, 20, 400),
]

def ai_controller(game_state):
    # Use the AI agent to make decisions based on the game state
    ai_agent = AI_Agent()
    ai_actions = ai_agent.play_game(game_state)
    return ai_actions

def death():
    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    player_speed = 0
    player_direction = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    game_state = {
        "player_position": player_pos,
        "player_speed": player_speed,
        "player_direction": player_direction,
        "walls": walls,
    }

    ai_actions = ai_controller(game_state)

    if ai_actions["forward"]:
        player_speed += ACCELERATION * dt
    if ai_actions["steer_left"]:
        player_change_angle = -STEERING_SPEED * dt
    if ai_actions["steer_right"]:
        player_change_angle = STEERING_SPEED * dt

    player_direction += player_change_angle
    player_speed = min(MAX_SPEED, player_speed)
    player_speed = max(-MAX_SPEED, player_speed)

    player_pos.x += player_speed * math.sin(player_direction)
    player_pos.y += player_speed * math.cos(player_direction)

    for wall in walls:
        if pygame.Rect(
            player_pos[0] - player_radius,
            player_pos[1] - player_radius,
            2 * player_radius,
            2 * player_radius,
        ).colliderect(wall):
            death()

    screen.fill("white")

    for wall in walls:
        pygame.draw.rect(screen, (0, 0, 0), wall)

    rotated_player = pygame.transform.rotate(
        player_image, math.degrees(player_direction)
    )
    player_rect = rotated_player.get_rect(center=player_pos)
    screen.blit(rotated_player, player_rect.topleft)

    pygame.display.flip()

    dt = clock.tick(60) / 1000