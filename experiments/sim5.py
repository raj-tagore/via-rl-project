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

spring_anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
spring_anchor.position = (100, 500)
space.add(spring_anchor)

surface = pymunk.Body(1, 1)
surface.position = (100, 400)
surface_shape = pymunk.Segment(surface, (-100, 0), (100, 0), 5)
space.add(surface, surface_shape)

spring = pymunk.DampedSpring(spring_anchor, surface, (0, 0), (0, 0), 100, 100, 0)
space.add(spring)

robot_torso = pymunk.Body(1, 1)
robot_torso.position = (100, 250)
robot_torso_shape = pymunk.Poly.create_box(robot_torso, (50, 50))
space.add(robot_torso, robot_torso_shape)

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