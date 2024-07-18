import pymunk
import pymunk.pygame_util
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((200, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

space = pymunk.Space()
space.gravity = (0, 900)

fixed_body = pymunk.Body(body_type=pymunk.Body.STATIC)
fixed_body.position = (100, 500)
fixed_body_shape = pymunk.Segment(fixed_body, (-100, 0), (100, 0), 5)
space.add(fixed_body, fixed_body_shape)

surface_body = pymunk.Body(1, 10)
surface_body.position = (100, 400)
surface_body_shape = pymunk.Segment(surface_body, (-100, 0), (100, 0), 5)
space.add(surface_body, surface_body_shape)

spring = pymunk.DampedSpring(fixed_body, surface_body, (0, 0), (0, 0), 100, 100, 5)
space.add(spring)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
    space.step(1/60)
    screen.fill((255, 255, 255))
    space.debug_draw(draw_options)
    pygame.display.update()
    clock.tick(60)