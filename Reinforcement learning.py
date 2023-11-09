import copy
import math
import sys
import pygame
import torch
import torch.nn as nn
import pygame.font
import threading
from queue import Queue

# constants

# game settings
SPRITE_SCALING = 0.075
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN_TITLE = "RL Tank"
ACCELERATION = 0.1
STEERING_SPEED = 0.1
MIN_SPEED = 0
MAX_SPEED = 100
INITIAL_POS = 50, 50
IMAGE_PATH = "tank_real_2.png"
RAYCAST_STEP_SIZE = 10
FRAMES_PER_SECOND = 60
WALLS = [
    pygame.Rect(300, 0, 20, 600),
    pygame.Rect(600, 200, 20, 600),
    pygame.Rect(900, 0, 20, 600),
]
TARGET = pygame.Rect(920, 0, 300, 30)


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
PERTURBATION_SCALES = [
    0.25,
    0.25,
    0.25,
    0.25,
    0.1,
    0.1,
    0.1,
    0.1,
    0.05,
    0.05,
    0.025,
    0.025,
    0.01,
    0.01,
    0.005,
    0.0025,
]
# the last value is never used

MODEL_PATH = "saved networks/Reinforcement learning.pth"
FONT_SIZE = 36
render_deaths = False
render_hitbox = False
render_raycasts = False

# TODO
"""
belangrijke todos:
    score systeem verbeteren:
        checkpoint systeem
        A* pathfinding
    met de GPU werken
    beste score onthouden ipv beste network onthouden en beste score telkens opnieuw te berekenen

andere route maken

render uit/aanknop op scherm/ een bepaalde toets om renderen te toggelen?

ipv random pertubations, meer nadenken over wat er veranderd moet worden: onthouden wat er al geprobeerd is (gradient descent?)
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                global render_deaths
                render_deaths = not render_deaths
            elif event.key == pygame.K_h:
                global render_hitbox
                render_hitbox = not render_hitbox
            elif event.key == pygame.K_r:
                global render_raycasts
                render_raycasts = not render_raycasts


def render_text(batch):
    pygame.draw.rect(screen, (158, 158, 232), pygame.Rect(0, 0, 300, 40))
    text = font.render(
        f"Current Batch: {batch}", True, (255, 255, 255)  # Text color (white)
    )
    text_rect = text.get_rect()
    text_rect.topleft = (10, 10)  # Position of the text on the screen
    screen.blit(text, text_rect)


def render(players, walls, should_raycasts, batch):
    event_handling()
    # clear previous screen
    screen.fill((200, 200, 255))

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
        if render_hitbox:
            pygame.draw.rect(screen, (0, 255, 0), player.rect, 1)
        p_index += 1

    # Render text
    render_text(batch)

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


def run(player, network, pos_in_batch, score_queue):
    running = True
    step = 0  # aka frame
    score = 0
    acceleration = 0
    change_angle = 0

    while running:
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
            score = calculate_score(
                player.pos.x, player.pos.y, step, player.rect.colliderect(TARGET)
            )

        # Get inputs for the neural network
        inputs = torch.tensor(player.get_inputs(), dtype=torch.float32)

        # Forward pass
        outputs = network(inputs)

        # Extract steering direction and acceleration from the outputs
        # and squach to [-1, 1]
        change_angle = torch.tanh(outputs[0]).item()
        acceleration = torch.tanh(outputs[1]).item()
    # render a red circle on death location
    if render_deaths:
        pygame.draw.circle(screen, (((pos_in_batch + 1) * 16) - 1, 0, 0), player.pos, 3)
    score_queue.put({pos_in_batch: score})


def render_run(player, network, batch):
    running = True
    step = 0  # aka frame
    acceleration = 0
    change_angle = 0

    while running:
        # game logic
        player.update(change_angle, acceleration)
        running = not player.check_collision()

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
        render([player], WALLS, [render_raycasts], batch)


def perturb_model(model, perturbation_scale):
    returnable_model = copy.deepcopy(model)
    for param in returnable_model.parameters():
        perturbation = perturbation_scale * torch.randn_like(param)
        param.data.add_(perturbation)
    return returnable_model


def find_value(data_list, pos):
    for data in data_list:
        key = list(data.keys())[0]
        if key == pos:
            return list(data.values())[0]


def extract_and_sort_values(data_list):
    data_dict = {}
    for i in range(len(data_list)):
        data_dict[i] = find_value(data_list, i)
    returnable = []
    for i in range(len(data_dict)):
        returnable.append(data_dict[i])
    return returnable


def train_batch(networks, current_batch):
    scores = []
    threads = []
    score_queue = Queue()
    for i in range(len(networks)):
        thread = threading.Thread(
            target=run, args=(Player(), networks[i], i, score_queue)
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    while not score_queue.empty():
        scores.append(score_queue.get())
    scores = extract_and_sort_values(scores)
    render_text(current_batch)
    pygame.display.flip()
    return scores


# render without player
render([], WALLS, [], "0")

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

scores = train_batch(networks, 0)

best_score = max(scores)
highest_score_index = scores.index(best_score)

render_run_thread = threading.Thread(
    target=render_run, args=(Player(), networks[highest_score_index], "0")
)
render_run_thread.start()

# clear terminal
print("\033c", end="")
# print info on current batch
print("current batch: 0\nlast change: n/a\npoints: " + str(best_score))

best_network = networks[highest_score_index]
last_batch_change = "n/a"

# perform all runs
for batch in range(BATCH_COUNT):
    event_handling()
    networks = [best_network]

    # copy and perturbate the previous best
    for j in range(BATCH_SIZE - 1):
        network = perturb_model(best_network, PERTURBATION_SCALES[j])
        networks.append(network)

    # train new batch
    scores = train_batch(networks, batch + 1)

    highest_score_index = scores.index(max(scores))
    if scores[highest_score_index] > best_score:
        best_score = scores[highest_score_index]
        last_batch_change = "batch " + str(batch + 1)

        print(
            "\033[92mnew best, look at game window to see, index: "
            + str(highest_score_index)
            + "\033[0m"
        )

        best_network = networks[highest_score_index]
        # save best network to file
        torch.save(best_network.state_dict(), MODEL_PATH)
        # render the new best
        render_run_thread.join()
        render_run_thread = threading.Thread(
            target=render_run, args=(Player(), best_network, batch + 1)
        )
        render_run_thread.start()

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
