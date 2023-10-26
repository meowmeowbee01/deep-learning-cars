import sys
import math
import pygame

# Constants
SPRITE_SCALING = 0.05
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 700
SCREEN_TITLE = "game"
ACCELERATION = 1
STEERING_SPEED = 2.5
MAX_SPEED = 20
INITIAL_POS = 60, 350


class Player:
    def __init__(
        self, image_path, initial_pos, acceleration, steering_speed, max_speed
    ):
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
        self.radius = math.sqrt(
            (self.image.get_width() / 2) ** 2 + (self.image.get_height() / 2) ** 2
        )
        self.rect = self.image.get_rect(center=self.pos)
        self.acceleration = acceleration
        self.steering_speed = steering_speed
        self.max_speed = max_speed

    def update(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and not keys[pygame.K_s]:
            self.speed += self.acceleration * dt
        elif keys[pygame.K_s] and not keys[pygame.K_w]:
            self.speed -= self.acceleration * dt

        if keys[pygame.K_a] and not keys[pygame.K_d]:
            self.change_angle = self.steering_speed * dt
        elif keys[pygame.K_d] and not keys[pygame.K_a]:
            self.change_angle = -self.steering_speed * dt
        elif not keys[pygame.K_d] and not keys[pygame.K_a]:
            self.change_angle = 0
        elif keys[pygame.K_d] and keys[pygame.K_a]:
            self.change_angle = 0

        self.direction += self.change_angle
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
                    return x, y

        return x, y

    def check_collision(self, walls):
        for wall in walls:
            if pygame.Rect(
                self.pos[0] - self.radius,
                self.pos[1] - self.radius,
                2 * self.radius,
                2 * self.radius,
            ).colliderect(wall):
                self.death()

        if (
            self.pos.x < 0 + self.radius
            or self.pos.x > SCREEN_WIDTH - self.radius
            or self.pos.y < 0 + self.radius
            or self.pos.y > SCREEN_HEIGHT - self.radius
        ):
            self.death()

    def death(self):
        self.pos = pygame.Vector2(INITIAL_POS)
        self.speed = 0
        self.direction = 0


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)
clock = pygame.time.Clock()
dt = 0

# Wall setup
walls = [
    pygame.Rect(0, 300, 150, 20),
    pygame.Rect(130, 320, 20, 100),
    pygame.Rect(0, 500, 300, 20),
]

# Create player instance
player = Player(
    "tank_real_2.png",
    INITIAL_POS,
    ACCELERATION,
    STEERING_SPEED,
    MAX_SPEED,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    player.update(dt)
    player.check_collision(walls)

    screen.fill("white")
    for wall in walls:
        pygame.draw.rect(screen, (0, 0, 0), wall)

    for angle in range(0, 181, 45):
        ray_angle = math.radians(angle) - player.direction
        end_point = player.cast_ray(ray_angle)
        pygame.draw.circle(screen, (0, 255, 0), end_point, 5)
        pygame.draw.line(
            screen, (255, 127, 127), player.pos, (int(end_point[0]), int(end_point[1]))
        )

    rotated_player = pygame.transform.rotate(
        player.image, math.degrees(player.direction)
    )
    player.rect = rotated_player.get_rect(center=player.pos)
    screen.blit(rotated_player, player.rect.topleft)

    pygame.display.flip()
    dt = clock.tick(60) / 1000
