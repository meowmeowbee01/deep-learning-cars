import math
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
SPRITE_SCALING = 0.05
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
SCREEN_DIAGONAL = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
SCREEN_TITLE = "Deep Learning Tank"
ACCELERATION = 0.2
STEERING_SPEED = 0.1
MAX_SPEED = 5
INITIAL_POS = 60, 350

# Training parameters
input_size = 9  # speed, direction, position(x, y), distance from 5 raycasts
output_size = 2  # steering direction, acceleration
learning_rate = 0.001
epochs = 1000


class Player:
    def __init__(self, image_path, initial_pos, max_speed):
        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(
            self.image,
            (
                int(self.image.get_width() * SPRITE_SCALING),
                int(self.image.get_height() * SPRITE_SCALING),
            ),
        )
        self.pos = pygame.Vector2(initial_pos)
        self.speed = 0
        self.direction = 0  # radians
        self.change_angle = 0
        self.acceleration = 0
        self.radius = math.sqrt(
            (self.image.get_width() / 2) ** 2 + (self.image.get_height() / 2) ** 2
        )
        self.rect = self.image.get_rect(center=self.pos)
        self.max_speed = max_speed

        self.raycast_hits = []

    def update(self, dt):
        self.direction += self.change_angle * STEERING_SPEED

        self.speed += self.acceleration * ACCELERATION

        self.speed = min(self.max_speed, self.speed)
        self.speed = max(-self.max_speed, self.speed)

        self.pos.x += self.speed * math.sin(self.direction)
        self.pos.y += self.speed * math.cos(self.direction)

    def cast_ray(self, angle):
        x, y = self.pos
        step_size = 5

        while 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

            for wall in walls:
                if wall.collidepoint(x, y):
                    return pygame.Vector2(x, y)

        return pygame.Vector2(x, y)

    def cast_rays(self):
        self.raycast_hits = []
        for angle in range(0, 181, 45):
            ray_angle = math.radians(angle) - player.direction
            end_point = self.cast_ray(ray_angle)
            self.raycast_hits.append(end_point)

    # returns True if death was triggered, and the distance to the right
    def check_collision(self, walls):
        for wall in walls:
            if pygame.Rect(
                self.pos[0] - self.radius,
                self.pos[1] - self.radius,
                2 * self.radius,
                2 * self.radius,
            ).colliderect(wall):
                self.death()
                return True, self.pos.x

        if (
            self.pos.x < 0 + self.radius
            or self.pos.x > SCREEN_WIDTH - self.radius
            or self.pos.y < 0 + self.radius
            or self.pos.y > SCREEN_HEIGHT - self.radius
        ):
            self.death()
            return True, self.pos.x

        return False, self.pos.x

    def death(self):
        self.pos = pygame.Vector2(INITIAL_POS)
        self.speed = 0
        self.direction = 0

    def get_inputs(self):
        return [
            self.speed / MAX_SPEED,  # Normalize speed to [0, 1]
            self.direction * (2 * math.pi),  # Normalize direction to [0, 1]
            self.pos.x / SCREEN_WIDTH,  # Normalize x position to [0, 1]
            self.pos.y / SCREEN_HEIGHT,  # Normalize y position to [0, 1]
            pygame.Vector2(self.pos - self.raycast_hits[0]).length() / SCREEN_DIAGONAL,
            pygame.Vector2(self.pos - self.raycast_hits[1]).length() / SCREEN_DIAGONAL,
            pygame.Vector2(self.pos - self.raycast_hits[2]).length() / SCREEN_DIAGONAL,
            pygame.Vector2(self.pos - self.raycast_hits[3]).length() / SCREEN_DIAGONAL,
            pygame.Vector2(self.pos - self.raycast_hits[4]).length() / SCREEN_DIAGONAL,
        ]


# Neural Network Definition
class CarController(nn.Module):
    def __init__(self, input_size, output_size):
        super(CarController, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# neural net instance
model = CarController(input_size, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()
dt = 0

# Wall setup
walls = [pygame.Rect(0, 310, 150, 20), pygame.Rect(0, 500, 300, 20)]

# Create player instance
player = Player(
    "tank_real_2.png",
    INITIAL_POS,
    MAX_SPEED,
)

# Training loop
for epoch in range(epochs):
    running = True
    total_reward = 0
    i = 0
    while running:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # game logic
        player.update(dt)
        player.cast_rays()
        died, reward = player.check_collision(walls)

        if i > 5 * 60:
            died = True
            player.death()

        running = not died

        total_reward += reward

        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = model(inputs)

        # Extract steering direction and acceleration from the outputs
        player.change_angle = torch.tanh(outputs[0]).item()
        player.acceleration = torch.tanh(outputs[1]).item()

        # print(str(player.change_angle) + "\t" + str(player.acceleration))

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
        dt = clock.tick(60) / 1000

    # Backpropagation and optimization
    optimizer.zero_grad()
    reward_tensor = torch.tensor([total_reward] * outputs.size(0), dtype=torch.float32)
    loss = criterion(outputs, reward_tensor)
    loss.backward()
    optimizer.step()
