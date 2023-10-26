import copy
import math
import sys
import pygame
import torch
import torch.nn as nn
import pygame.font

# constants

# game settings
SPRITE_SCALING = 0.05
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SCREEN_TITLE = "RL Tank"
ACCELERATION = 0.2
STEERING_SPEED = 0.25
MIN_SPEED = 0
MAX_SPEED = 10
INITIAL_POS = 50, 50
IMAGE_PATH = "tank_real_2.png"
RAYCAST_STEP_SIZE = 5
FRAMES_PER_SECOND = 60
WALLS = [
    pygame.Rect(300, 0, 20, 350),
    pygame.Rect(600, 350, 20, 350),
    pygame.Rect(900, 0, 20, 350),
]
TARGET = 1150, 50

# hyperparameters
INPUT_SIZE = 7  # speed, direction, 5 raycasts
OUTPUT_SIZE = 2  # acceleration, steering
HIDDEN_LAYER_SIZE = 64

# training settings
TIME_PER_RUN = 15  # seconds
BATCH_COUNT = 250  # an extra 0th batch will always happen first
# the first position in a batch is be reserved for the previous best
# before the 0th run, the previous best will be pulled from a file
BATCH_SIZE = 8
PERTURBATION_SCALE = 0.03


MODEL_PATH = "saved networks/Reinforcement learning.pth"
SHOULD_RENDER = False

# TODO
"""
code opruimen

beste score onthouden ipv beste network onthouden en beste score telkens opnieuw te berekenen
render functie aanpassen om compatibel te zijn met meerdere players

Async -> grafische kaart -> meerdere runnen in batch
belangrijk: kleurtjes (andere kleur voor record)

beter scoresysteem -> pathfinding
andere route maken

render uit/aanknop op scherm?

andere PERTURBATION_SCALE per network in een batch?
"""


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

    def update(self, deltaTime):
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

        # update rect
        rotated_player = pygame.transform.rotate(
            self.image, math.degrees(self.direction)
        )
        self.rect = rotated_player.get_rect(center=self.pos)

    # returns Vector2 of the absolute position of the hit point
    def cast_ray(self, angle):
        x, y = self.pos
        step_size = RAYCAST_STEP_SIZE

        while 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

            for wall in WALLS:
                if wall.collidepoint(x, y):
                    return pygame.Vector2(x, y)

        return pygame.Vector2(x, y)

    def cast_rays(self):
        self.raycast_hits = []
        # 5 raycasts, 45 degree seperation: 0, 45, 90, 135, 180
        for angle in range(0, 181, 45):
            ray_angle = math.radians(angle) - self.direction
            hit = self.cast_ray(ray_angle)
            self.raycast_hits.append(hit)

    # returns True if wall was hit or oob
    def check_collision(self):
        for wall in WALLS:
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
            # then squach to maximum 1
            squached_distance = math.tanh(
                distance / ((SCREEN_HEIGHT + SCREEN_WIDTH) / 2)
            )
            # add to inputs
            inputs.append(squached_distance)

        for value in inputs:
            if value > 1 or value < 0:
                raise Exception("inputs not normalized: " + value)
        return inputs


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()

pygame.font.init()
font = pygame.font.Font(None, 36)

current_batch = 0


def render(player):
    # clear previous screen
    screen.fill((127, 127, 127))

    # render walls
    for wall in WALLS:
        pygame.draw.rect(screen, (0, 0, 0), wall)

    # render raycast
    for hit in player.raycast_hits:
        pygame.draw.circle(screen, (0, 127, 0), hit, 5)
        pygame.draw.line(screen, (127, 0, 0), player.pos, (int(hit[0]), int(hit[1])))

    # render player
    rotated_player = pygame.transform.rotate(
        player.image, math.degrees(player.direction)
    )
    screen.blit(rotated_player, player.rect.topleft)

    # Render text (now text is on top of other elements)
    text = font.render(
        f"Current Batch: {current_batch}", True, (255, 255, 255)  # Text color (white)
    )
    text_rect = text.get_rect()
    text_rect.topleft = (10, 10)  # Position of the text on the screen
    screen.blit(text, text_rect)

    # output render to display and update deltatime
    pygame.display.flip()
    deltaTime = clock.tick(FRAMES_PER_SECOND) / 1000  # milliseconds

    return deltaTime


def run(player, network, shouldRender=False):
    running = True
    deltaTime = 0
    i = 0

    score = 0
    while running:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # game logic
        player.update(deltaTime)
        died = player.check_collision()
        running = not died

        # stop after too much time
        if i > TIME_PER_RUN * FRAMES_PER_SECOND:
            running = False
        i += 1

        if not running:
            if player.pos.x < 310 or (player.pos.x > 610 and player.pos.x < 910):
                score = (player.pos.x * 5) + player.pos.y
            else:
                score = (player.pos.x * 5) + SCREEN_HEIGHT - player.pos.y
                if player.pos.x > 910:
                    score += -(TIME_PER_RUN * FRAMES_PER_SECOND - i) / 5
        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = network(inputs)

        # Extract steering direction and acceleration from the outputs
        # and squach to [-1, 1]
        player.change_angle = torch.tanh(outputs[0]).item()
        player.acceleration = torch.tanh(outputs[1]).item()

        # rendering
        if shouldRender:
            deltaTime = render(player)
    return score


def perturb_model(model, perturbation_scale):
    returnable_model = copy.deepcopy(model)
    for param in returnable_model.parameters():
        perturbation = perturbation_scale * torch.randn_like(param)
        param.data.add_(perturbation)
    return returnable_model


def train_batch(networks):
    scores = []
    for network in networks:
        score = run(Player(), network, SHOULD_RENDER)
        scores.append(score)
    return scores


# load previous best from file
network = Network()
try:
    network.load_state_dict(torch.load(MODEL_PATH))
    networks = [network]
except:
    networks = []

# run 0
for i in range(BATCH_SIZE - 1):
    networks.append(Network())

scores = train_batch(networks)

highest_score_index = scores.index(max(scores))
print(
    "batch 0: best run: "
    + str(highest_score_index + 1)
    + " with "
    + str(scores[highest_score_index])
    + " points"
)

best_network = networks[highest_score_index]
best_score = scores[highest_score_index]
last_batch_change = 0
# perform all runs
for i in range(BATCH_COUNT):
    networks = [best_network]

    # copy and perturbate the previous best
    for j in range(BATCH_SIZE - 1):
        network = perturb_model(best_network, PERTURBATION_SCALE)
        networks.append(network)

    # train new generation
    scores = train_batch(networks)

    highest_score_index = scores.index(max(scores))
    if scores[highest_score_index] > best_score:
        best_score = scores[highest_score_index]
        last_batch_change = i + 1
        run(Player(), best_network, True)

    print("\033c", end="")
    print(
        "current batch: "
        + str(i + 1)
        + "\nlast change: batch "
        + str(last_batch_change)
        + "\npoints: "
        + str(scores[highest_score_index])
    )
    current_batch = i + 2

    best_network = networks[highest_score_index]
    torch.save(best_network.state_dict(), MODEL_PATH)
