import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Create a space
space = pymunk.Space()
space.gravity = (0, 900)  # Gravity (x, y)

torso_body = pymunk.Body(body_type=pymunk.Body.STATIC)
torso_body.position = (400, 400)
torso_shape = pymunk.Circle(torso_body, 50)
torso_shape.elasticity = 0.4
torso_shape.friction = 0.5
space.add(torso_body, torso_shape)

# Create the limb
mass_limb = 5
size_limb = (100, 20)
moment_limb = pymunk.moment_for_box(mass_limb, size_limb)

limb_body = pymunk.Body(mass_limb, moment_limb)
limb_body.position = (500, 400)
limb_shape = pymunk.Poly.create_box(limb_body, size_limb)
limb_shape.elasticity = 0.4
limb_shape.friction = 0.5
space.add(limb_body, limb_shape)

# Create a PivotJoint to connect the limb to the center of the circular body
pivot_joint = pymunk.PivotJoint(torso_body, limb_body, (400, 400))
space.add(pivot_joint)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill((255, 255, 255))  # White background
    
    # Step the simulation
    space.step(1/60.0)
    
    # Draw the space
    space.debug_draw(draw_options)
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)
