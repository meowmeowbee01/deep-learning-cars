import math
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Constants

# settings
SPRITE_SCALING = 0.05
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Deep Learning Tank"
ACCELERATION = 0.1
STEERING_SPEED = 0.1
MAX_SPEED = 10
INITIAL_POS = 60, 350
IMAGE_PATH = "tank_real_2.png"
RAYCAST_STEP_SIZE = 5
FRAMES_PER_SECOND = 60
TIME_PER_RUN = 5  # seconds

# Training parameters
INPUT_SIZE = 7  # speed, direction, distance from 5 raycasts
HIDDEN_LAYER_SIZE = 8
OUTPUT_SIZE = 2  # steering direction, acceleration
LEARNING_RATE = 0.01
EPOCHS = 1000


class Player:
    def __init__(self):
        print("new player instance")
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
        # update speed
        self.speed += self.acceleration * ACCELERATION

        # cap speed
        self.speed = min(MAX_SPEED, self.speed)
        self.speed = max(-MAX_SPEED, self.speed)

        # update position
        self.pos.x += self.speed * math.sin(self.direction)
        self.pos.y += self.speed * math.cos(self.direction)

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
        # 5 raycasts, 45 degree seperation: 0, 45, 90, 135, 180
        for angle in range(0, 181, 45):
            ray_angle = math.radians(angle) - player.direction
            hit = self.cast_ray(ray_angle)
            self.raycast_hits.append(hit)

    # returns True if wall was hit
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
        return [
            self.speed / MAX_SPEED,  # Normalize speed to [0, 1]
            self.direction * 2 * math.pi,  # Normalize direction to [0, 1]
            pygame.Vector2(self.pos - self.raycast_hits[0]).length(),  # not normalized
            pygame.Vector2(self.pos - self.raycast_hits[1]).length(),  # not normalized
            pygame.Vector2(self.pos - self.raycast_hits[2]).length(),  # not normalized
            pygame.Vector2(self.pos - self.raycast_hits[3]).length(),  # not normalized
            pygame.Vector2(self.pos - self.raycast_hits[4]).length(),  # not normalized
        ]


# Neural Network Definition
class CarController(nn.Module):
    def __init__(self):
        super(CarController, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# neural net instance
model = CarController()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()

# Training loop
for epoch in range(EPOCHS):
    running = True
    total_reward = 0
    i = 0
    dt = 0

    # Wall setup
    walls = [
        pygame.Rect(0, 200, 700, 20),
        pygame.Rect(0, 600, 700, 20),
        pygame.Rect(200, 200, 20, 150),
    ]

    # Create player instance
    player = Player()

    while running:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # game logic
        player.update(dt)
        player.cast_rays()
        died = player.check_collision(walls)

        # die after too much time
        if i > TIME_PER_RUN * FRAMES_PER_SECOND:
            died = True
        i += 1

        running = not died

        reward = player.pos.x
        total_reward += reward

        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = model(inputs)

        # Extract steering direction and acceleration from the outputs
        # and normalize to [-1, 1]
        player.change_angle = torch.tanh(outputs[0]).item()
        player.acceleration = torch.tanh(outputs[1]).item()

        # rendering

        # clear previous screen
        screen.fill((127, 127, 127))

        # render walls
        for wall in walls:
            pygame.draw.rect(screen, (0, 0, 0), wall)

        # render raycast
        for hit in player.raycast_hits:
            pygame.draw.circle(screen, (0, 255, 0), hit, 5)
            pygame.draw.line(
                screen, (255, 127, 127), player.pos, (int(hit[0]), int(hit[1]))
            )

        # render player
        rotated_player = pygame.transform.rotate(
            player.image, math.degrees(player.direction)
        )
        player.rect = rotated_player.get_rect(center=player.pos)
        screen.blit(rotated_player, player.rect.topleft)

        # output render to display and update deltatime
        pygame.display.flip()
        dt = clock.tick(60) / 1000  # milliseconds

    # Backpropagation and optimization
    optimizer.zero_grad()
    reward_tensor = torch.tensor([total_reward] * outputs.size(0), dtype=torch.float32)
    loss = criterion(outputs, reward_tensor)
    loss.backward()
    optimizer.step()
