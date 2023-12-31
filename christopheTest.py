import copy
import math
import sys
import pygame
import torch
import torch.nn as nn
import pygame.font

# constants

# game settings
SPRITE_SCALING = 0.125
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SCREEN_TITLE = "RL Tank"
ACCELERATION = 0.2
STEERING_SPEED = 0.1
MIN_SPEED = 0
MAX_SPEED = 100
INITIAL_POS = 50, 50
IMAGE_PATH = "tank_real_2.png"
RAYCAST_STEP_SIZE = 5
FRAMES_PER_SECOND = 60
WALLS = [
    #horizontaal
    pygame.Rect(300, 0, 20, 350),
    pygame.Rect(600, 350, 20, 350),
    pygame.Rect(900, 0, 20, 350),

    #verticaal
    pygame.Rect(150, 450, 300, 20),
    pygame.Rect(450, 250, 300, 20),
    pygame.Rect(750, 450, 300, 20),

]
TARGET = (pygame.Rect(910, 0, 300, 30),)


# hyperparameters
INPUT_SIZE = 7  # speed, direction, 5 raycasts
OUTPUT_SIZE = 2  # acceleration, steering
FIRST_HIDDEN_LAYER_SIZE = 8
SECOND_HIDDEN_LAYER_SIZE = 8

# training settings
TIME_PER_RUN = 10  # seconds
BATCH_COUNT = 1000  # an extra 0th batch will always happen first
# the first position in a batch is reserved for the previous best
# before the 0th run, the previous best will be pulled from a file
BATCH_SIZE = 16
PERTURBATION_SCALES = [0.5, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001, 0.000005, 0.0000025, 0.000001]
# the last value is never used

MODEL_PATH = "saved networks/Reinforcement learning.pth"
FONT_SIZE = 36
RENDER_DEATHS = True

# TODO
"""
belangrijke todos:
    score systeem verbeteren:
        checkpoint systeem
        A* payhfinding
    async maken:
        render en input handling op een andere manier regelen
        meerdere runnen in een batch
        met de GPU werken

beste score onthouden ipv beste network onthouden en beste score telkens opnieuw te berekenen

belangrijk: kleurtjes (andere kleur voor record)

andere route maken

render uit/aanknop op scherm/ een bepaalde toets om renderen te toggelen?

ipv random pertubations, meer nadenken over wat er veranderd moet worden: onthouden wat er al geprobeerd is (gradient descent?)

experimenteren met hyperparameters: meer hidden layers, kleinere of grotere hidden layers, meer of minder raycasts
experimenteren met training parameters: perturbation scale groter of kleiner, grotere of kleinere batches
"""


def distance_between_2_points(p1, p2):
    rel_vector = pygame.Vector2(p1 - p2)
    distance = rel_vector.length()
    return distance


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
        self.raycast_hits = []

        self.radius = math.sqrt(
            (self.image.get_width() / 2) ** 2 + (self.image.get_height() / 2) ** 2
        )
        self.rect = self.image.get_rect(center=self.pos)

    def update(self, change_angle=0, acceleration=0):
        # update direction
        self.direction += change_angle * STEERING_SPEED

        # normalize direction to always be positive
        if self.direction <= 0:
            self.direction += math.radians(360)

        # normalize direction for full rotations
        self.direction = self.direction % (2 * math.pi)

        # update speed
        self.speed += acceleration * ACCELERATION

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
            (self.speed - MIN_SPEED)
            / (MAX_SPEED - MIN_SPEED),  # Normalize speed to [0, 1]
            self.direction / (2 * math.pi),  # Normalize direction to [0, 1]
        ]
        for hit in self.raycast_hits:
            distance = distance_between_2_points(self.pos, hit)
            # then squach to maximum 1
            squached_distance = math.tanh(
                distance / ((SCREEN_HEIGHT + SCREEN_WIDTH) / 2)
            )
            # add to inputs
            inputs.append(squached_distance)

        # check for inputs outside of [0, 1]
        # for value in inputs:
        #     if value > 1 or value < 0:
        #         raise Exception("inputs not normalized: " + value)
        return inputs


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, FIRST_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(SECOND_HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()

pygame.font.init()
font = pygame.font.Font(None, FONT_SIZE)


def event_handling():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def render(players, walls, should_raycasts, batch):
    # clear previous screen
    screen.fill((127, 127, 127))

    # render target
    pygame.draw.rect(screen, (255, 64, 128), TARGET)

    # render walls
    for wall in walls:
        pygame.draw.rect(screen, (0, 0, 0), wall)

    p_index = 0
    for player in players:
        # render raycast
        if should_raycasts[p_index]:
            for hit in player.raycast_hits:
                pygame.draw.circle(screen, (0, 127, 0), hit, 5)
                pygame.draw.line(
                    screen, (127, 0, 0), player.pos, (int(hit[0]), int(hit[1]))
                )

        # render player
        rotated_player = pygame.transform.rotate(
            player.image, math.degrees(player.direction)
        )
        screen.blit(rotated_player, player.rect.topleft)
        pygame.draw.rect(screen, (0, 255, 0), player.rect, 1)
        p_index += 1

    # Render text
    text = font.render(
        f"Current Batch: {batch}", True, (255, 255, 255)  # Text color (white)
    )
    text_rect = text.get_rect()
    text_rect.topleft = (10, 10)  # Position of the text on the screen
    screen.blit(text, text_rect)

    # output render to display
    pygame.display.flip()
    clock.tick(FRAMES_PER_SECOND)

def calculate_score(x, y, step, hit_finish):
    if x < 310:
        score = x + y
    elif x < 610:
        score = x + SCREEN_HEIGHT - y + 1000
    elif x < 910:
        score = x + y + 2000
    else:
        score = SCREEN_HEIGHT - y + 5000

    # bonus score: faster to target -> more points
    if hit_finish:
        score += TIME_PER_RUN * FRAMES_PER_SECOND - step

    return score


def run(player, network, pos_in_batch):
    running = True
    step = 0  # aka frame
    score = 0
    acceleration = 0
    change_angle = 0

    while running:
        # this loop is the most nested loop so we do event handling here, despite it seeming out of place
        event_handling()

        # game logic
        player.update(change_angle, acceleration)
        died = player.check_collision()
        running = not died

        # stop after too much time
        if step > TIME_PER_RUN * FRAMES_PER_SECOND:
            running = False
        step += 1
        
        if died:
            score = calculate_score(player.pos.x, player.pos.y, step, player.rect.colliderect(TARGET))

        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = network(inputs)

        # Extract steering direction and acceleration from the outputs
        # and squach to [-1, 1]
        change_angle = torch.tanh(outputs[0]).item()
        acceleration = torch.tanh(outputs[1]).item()
    # render a red circle on death location
    if RENDER_DEATHS:
        pygame.draw.circle(screen, (((pos_in_batch + 1) * 16) - 1, 0, 0), player.pos, 3)
    return score


def render_run(player, network, batch):
    running = True
    step = 0  # aka frame
    acceleration = 0
    change_angle = 0

    while running:
        # this loop is the most nested loop so we do event handling here, despite it seeming out of place
        event_handling()

        # game logic
        player.update(change_angle, acceleration)
        died = player.check_collision()
        running = not died

        # stop after too much time
        if step > TIME_PER_RUN * FRAMES_PER_SECOND:
            running = False
        step += 1

        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = network(inputs)

        # Extract steering direction and acceleration from the outputs
        # and squach to [-1, 1]
        change_angle = torch.tanh(outputs[0]).item()
        acceleration = torch.tanh(outputs[1]).item()

        # rendering
        render([player], WALLS, [True], batch)


def perturb_model(model, perturbation_scale):
    returnable_model = copy.deepcopy(model)
    for param in returnable_model.parameters():
        perturbation = perturbation_scale * torch.randn_like(param)
        param.data.add_(perturbation)
    return returnable_model


def train_batch(networks):
    scores = []
    i = 0
    for network in networks:
        score = run(Player(), network, i)
        scores.append(score)
        if RENDER_DEATHS:
            pygame.display.flip()
        i += 1
    return scores


# render without player
render([], WALLS, False, "0")

# load previous best from file
pulled_from_file = False
networks = []
try:
    network = Network()
    network.load_state_dict(torch.load(MODEL_PATH))
    networks = [network]
    pulled_from_file = True
except:
    pass

# batch 0
for i in range(BATCH_SIZE - int(pulled_from_file)):
    networks.append(Network())

scores = train_batch(networks)

best_score = max(scores)
highest_score_index = scores.index(best_score)

render_run(Player(), networks[highest_score_index], "0")

# clear terminal
print("\033c", end="")
# print info on current batch
print("current batch: 0\nlast change: n/a\npoints: " + str(best_score))

best_network = networks[highest_score_index]
last_batch_change = "n/a"

# perform all runs
for batch in range(BATCH_COUNT):
    networks = [best_network]

    # copy and perturbate the previous best
    for j in range(BATCH_SIZE - 1):
        network = perturb_model(best_network, PERTURBATION_SCALES[j])
        networks.append(network)

    # train new batch
    scores = train_batch(networks)

    highest_score_index = scores.index(max(scores))
    if scores[highest_score_index] > best_score:
        best_score = scores[highest_score_index]
        last_batch_change = "batch " + str(batch + 1)

        print(
            "\033[92mnew best, look at game window to see, index: "
            + str(highest_score_index)
            + "\033[0m"
        )

        # render the new best
        render_run(Player(), best_network, batch + 1)

        # save best network to file
        best_network = networks[highest_score_index]
        torch.save(best_network.state_dict(), MODEL_PATH)

    # clear terminal
    print("\033c", end="")
    # print info on current batch
    print(
        "current batch: "
        + str(batch + 1)
        + "\nlast change: "
        + last_batch_change
        + "\npoints: "
        + str(best_score)
    )
