import pymunk as pm
import pymunk.pygame_util
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen) # defines screen on which to draw

space = pm.Space()
space.gravity = (0, 900)

floor_body = pm.Body(body_type=pm.Body.STATIC)
floor_shape = pm.Segment(floor_body, (0, 500), (600, 500), 5, elasticity=0.95)
space.add(floor_body, floor_shape)

body2 = pm.Body(1, 100)
body2.position = (300, 300)
shape2 = pm.Circle(body2, 50)
shape2.elasticity = 0.95
space.add(body2, shape2)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
    space.step(1/60)
    screen.fill((255, 255, 255))
    space.debug_draw(draw_options) # draws everything in pymunk space
    pygame.display.update()
    clock.tick(60)
