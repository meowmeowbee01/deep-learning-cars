import math
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import random

# Constants

# Game settings
SPRITE_SCALING = 0.05
SCREEN_WIDTH = 100
SCREEN_HEIGHT = 700
SCREEN_TITLE = "PPO RL Tank"
ACCELERATION = 0.1
STEERING_SPEED = 0.1
MIN_SPEED = 0
MAX_SPEED = 1
INITIAL_POS = 50, 50
IMAGE_PATH = "tank_real_2.png"
RAYCAST_STEP_SIZE = 5
FRAMES_PER_SECOND = 60
TIME_PER_RUN = 50  # seconds

# Training parameters
INPUT_SIZE = 7  # speed, direction, distance from 5 raycasts
HIDDEN_LAYER_SIZE = 64
OUTPUT_SIZE = 2  # steering direction, acceleration
LEARNING_RATE = 0.001
EPSILON = 0.02
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
NUM_EPOCHS = 100
NUM_MINI_BATCHES = 32
GAMMA = 0.2

# Value net parameters
VALUE_HIDDEN_LAYER_SIZE = 64

walls = [
    # pygame.Rect(0, 500, 300, 20),
]

torch.autograd.set_detect_anomaly(True)

class Player:
    def __init__(self):
        self.image = pygame.image.load(IMAGE_PATH).convert_alpha()
        self.image = pygame.transform.scale(
            self.image,
            (
                int(self.image.get_width() * SPRITE_SCALING),
                int(self.image.get_height() * SPRITE_SCALING),
            ),
        )
        self.pos = pygame.Vector2(INITIAL_POS)
        self.speed = 0
        self.direction = 0  # radians
        self.change_angle = 0
        self.acceleration = 0
        self.raycast_hits = []

        self.radius = math.sqrt(
            (self.image.get_width() / 2) ** 2 + (self.image.get_height() / 2) ** 2
        )
        self.rect = self.image.get_rect(center=self.pos)

    def update(self, dt):
        # update direction
        self.direction += self.change_angle * STEERING_SPEED

        # normalize direction to always be positive
        if self.direction <= 0:
            self.direction += math.radians(360)

        # normalize direction for full rotations
        self.direction = self.direction % (2 * math.pi)

        # update speed
        self.speed += self.acceleration * ACCELERATION

        # cap speed
        self.speed = min(MAX_SPEED, self.speed)
        self.speed = max(MIN_SPEED, self.speed)

        # update position
        self.pos.x += self.speed * math.sin(self.direction)
        self.pos.y += self.speed * math.cos(self.direction)

        # update raycasts
        self.cast_rays()

    # returns Vector2 of the absolute position of the hit point
    def cast_ray(self, angle):
        x, y = self.pos
        step_size = RAYCAST_STEP_SIZE

        while 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

            for wall in walls:
                if wall.collidepoint(x, y):
                    return pygame.Vector2(x, y)

        return pygame.Vector2(x, y)

    def cast_rays(self):
        self.raycast_hits = []
        # 5 raycasts, 45 degree separation: 0, 45, 90, 135, 180
        for angle in range(0, 181, 45):
            ray_angle = math.radians(angle) - self.direction
            hit = self.cast_ray(ray_angle)
            self.raycast_hits.append(hit)

    # returns True if a wall was hit or out of bounds
    def check_collision(self, walls):
        for wall in walls:
            if self.rect.colliderect(wall):
                return True

        if (
            self.pos.x < 0 + self.radius
            or self.pos.x > SCREEN_WIDTH - self.radius
            or self.pos.y < 0 + self.radius
            or self.pos.y > SCREEN_HEIGHT - self.radius
        ):
            return True

        return False

    def get_inputs(self):
        inputs = [
            self.speed / MAX_SPEED,  # Normalize speed to [0, 1]
            self.direction / (2 * math.pi),  # Normalize direction to [0, 1]
        ]
        for hit in self.raycast_hits:
            # calculate relative position of hit
            rel_pos = pygame.Vector2(self.pos - hit)
            # then calculate length
            distance = rel_pos.length()
            # then squash to a maximum of 1
            squashed_distance = math.tanh(
                distance / ((SCREEN_HEIGHT + SCREEN_WIDTH) / 2)
            )
            # add to inputs
            inputs.append(squashed_distance)

        for value in inputs:
            if value > 1 or value < 0:
                print("Inputs not normalized")
                print(value)
        return inputs

    def reset(self):
        self.pos = pygame.Vector2(INITIAL_POS)
        self.speed = 0
        self.direction = 0
        self.change_angle = 0
        self.acceleration = 0
        self.raycast_hits = []

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()  
        self.fc1 = nn.Linear(INPUT_SIZE, VALUE_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(VALUE_HIDDEN_LAYER_SIZE, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        state_value = self.fc2(x)
        return state_value


class PPOLoss(nn.Module):
    def __init__(self, clip_ratio, value_coefficient, entropy_coefficient):
        super(PPOLoss, self).__init__()
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

    def forward(self, advantages, old_log_probs, values, new_log_probs, clipped_values):
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        clipped_values = values + (clipped_values - values).clamp(
            -self.clip_ratio, self.clip_ratio
        )
        value_loss = (
            0.5
            * torch.max(
                (clipped_values - values).pow(2), (values - clipped_values).pow(2)
            ).mean()
        )

        # Entropy regularization
        entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coefficient * value_loss
            - self.entropy_coefficient * entropy
        )

        return total_loss

def calculate_gae_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    T = len(rewards)

    # Calculate temporal differences
    deltas = []
    for t in range(T - 1):
        delta_t = rewards[t] + gamma * values[t + 1] - values[t]
        deltas.append(delta_t)
    deltas.append(rewards[-1] - values[-1])  # Last time step

    # Calculate GAE advantages
    advantages = torch.zeros(T, dtype=torch.float32)
    advantage = 0
    for t in reversed(range(T)):
        advantage = advantage * gamma * lambda_ + deltas[t]
        advantages[t] = advantage

    return advantages

# Initialize your networks and optimizer
policy_net = Policy()
value_net = Value()
optimizer = optim.Adam(
    list(policy_net.parameters()) + list(value_net.parameters()), lr=LEARNING_RATE
)

ppo_loss = PPOLoss(CLIP_RATIO, VALUE_COEFFICIENT, ENTROPY_COEFFICIENT)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()

# Create multiple tanks
NUM_TANKS = 10
tanks = [Player() for _ in range(NUM_TANKS)]

# Lists to track the state of each tank
tank_states = [{"hit_wall": False} for _ in range(NUM_TANKS)]

# PPO optimization loop
for epoch in range(NUM_EPOCHS):
    running = True
    i = 0
    dt = 0

    # Initialize states, actions, rewards, and values for each tank
    states = [[] for _ in range(NUM_TANKS)]
    actions = [[] for _ in range(NUM_TANKS)]
    rewards = [[] for _ in range(NUM_TANKS)]
    values = [[] for _ in range(NUM_TANKS)]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update and check the state of each tank
        for tank_idx, tank in enumerate(tanks):
            if not tank_states[tank_idx]["hit_wall"]:
                tank.update(dt)
                died = tank.check_collision(walls)

                running = running and not died

                if i > TIME_PER_RUN * FRAMES_PER_SECOND:
                    running = False
                i += 1

                reward = tank.pos.y

                if died:
                    reward = -SCREEN_HEIGHT
                    tank_states[tank_idx]["hit_wall"] = True

                inputs = torch.tensor(tank.get_inputs(), dtype=torch.float32)

                outputs = policy_net(inputs)

                tank.change_angle = torch.tanh(outputs[0]).item()
                tank.acceleration = torch.tanh(outputs[1]).item()

                states[tank_idx].append(inputs)
                actions[tank_idx].append(
                    torch.tensor([tank.change_angle, tank.acceleration], dtype=torch.float32)
                )
                rewards[tank_idx].append(torch.tensor(reward))
                values[tank_idx].append(value_net(inputs))

        screen.fill((0, 0, 0))

        for wall in walls:
            pygame.draw.rect(screen, (127, 127, 127), wall)

        for tank in tanks:
            for hit in tank.raycast_hits:
                pygame.draw.circle(screen, (0, 127, 0), hit, 5)
                pygame.draw.line(
                    screen, (127, 0, 0), tank.pos, (int(hit[0]), int(hit[1]))
                )

            rotated_tank = pygame.transform.rotate(
                tank.image, math.degrees(tank.direction)
            )
            tank.rect = rotated_tank.get_rect(center=tank.pos)
            screen.blit(rotated_tank, tank.rect.topleft)

        pygame.display.flip()
        dt = clock.tick(60) / 1000

    # Check if all tanks have hit a wall
    if all(tank_state["hit_wall"] for tank_state in tank_states):
        # Reset all tanks and start a new episode
        for tank_idx, tank in enumerate(tanks):
            tank_states[tank_idx]["hit_wall"] = False
            tank.reset()

    # ... (rest of the PPO optimization loop)


