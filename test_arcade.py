import math
import arcade

SPRITE_SCALING = 0.5

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Deep Learning Cars"

ACCELERATION = 5
STEERING_SPEED = 250

MAX_SPEED = 5

UP = arcade.key.W
DOWN = arcade.key.S
LEFT = arcade.key.A
RIGHT = arcade.key.D


class Player(arcade.Sprite):
    """Player class"""

    def __init__(self, image, scale):
        """Set up the player"""

        # Call the parent init
        super().__init__(image, scale)

        # Create a variable to hold our speed. 'angle' is created by the parent
        self.speed = 0

    def update(self):
        # Convert angle in degrees to radians.
        angle_rad = math.radians(self.angle)

        # Rotate the car
        self.angle += self.change_angle  # *self.speed?

        # Use math to find our change based on our speed and angle
        self.center_x += -self.speed * math.sin(angle_rad)
        self.center_y += self.speed * math.cos(angle_rad)

        # check for out-of-bounds
        if self.left < 0:
            self.left = 0
        elif self.right > SCREEN_WIDTH - 1:
            self.right = SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - 1


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title):
        """
        Initializer
        """

        # Call the parent class initializer
        super().__init__(width, height, title)

        # Variables that will hold sprite lists
        self.player_list = None

        # Set up the player info
        self.player_sprite = None

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        # Set the background color
        arcade.set_background_color(arcade.color.WHITE)

    def setup(self):
        """Set up the game and initialize the variables."""

        # Sprite lists
        self.player_list = arcade.SpriteList()

        # Set up the player
        self.playecr_sprite = Player(
            ":resources:images/topdown_tanks/tank_blue.png", SPRITE_SCALING
        )
        self.player_sprite.center_x = SCREEN_WIDTH / 2
        self.player_sprite.center_y = SCREEN_HEIGHT / 2
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Draw all the sprites.
        self.player_list.draw()

    def on_update(self, delta_time):
        """Movement and game logic"""

        if self.up_pressed and not self.down_pressed:
            self.player_sprite.speed -= ACCELERATION * delta_time

        elif self.down_pressed and not self.up_pressed:
            self.player_sprite.speed += ACCELERATION * delta_time

        if self.left_pressed and not self.right_pressed:
            self.player_sprite.change_angle = STEERING_SPEED * delta_time

        elif self.right_pressed and not self.left_pressed:
            self.player_sprite.change_angle = -STEERING_SPEED * delta_time

        elif not self.right_pressed and not self.left_pressed:
            self.player_sprite.change_angle = 0

        elif self.right_pressed and self.left_pressed:
            self.player_sprite.change_angle = 0

        self.player_sprite.speed = min(MAX_SPEED, self.player_sprite.speed)
        self.player_sprite.speed = max(-MAX_SPEED, self.player_sprite.speed)

        # Move the player
        self.player_list.update()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed."""

        # If the player presses a key, update the speed
        if key == UP:
            self.up_pressed = True
        elif key == DOWN:
            self.down_pressed = True
        elif key == LEFT:
            self.left_pressed = True
        elif key == RIGHT:
            self.right_pressed = True

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key."""

        if key == UP:
            self.up_pressed = False
        elif key == DOWN:
            self.down_pressed = False
        elif key == LEFT:
            self.left_pressed = False
        elif key == RIGHT:
            self.right_pressed = False


def main():
    """Main function"""
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
