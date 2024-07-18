import pymunk
import pymunk.pygame_util
import pygame
import sys

# Here I added 2 grooves to keep the surface straight and one spring for surface dynamics

pygame.init()
screen = pygame.display.set_mode((300, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
space = pymunk.Space()
space.gravity = (0, 900)

spring_anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
spring_anchor_shape = pymunk.Poly.create_box(spring_anchor, (250, 50))
spring_anchor.position = (150, 500)
space.add(spring_anchor, spring_anchor_shape)

surface = pymunk.Body(1, pymunk.moment_for_box(1, (300, 5)))
surface.position = (150, 400)
surface_shape = pymunk.Segment(surface, (-150, 0), (150, 0), 5)
space.add(surface, surface_shape)

spring1 = pymunk.DampedSpring(spring_anchor, surface, (0, 0), (0, 0), 100, 100, 0)
space.add(spring1)

groove = pymunk.GrooveJoint(spring_anchor, surface, (-100, 0), (-100, -500), (-100, 0))
space.add(groove)
groove2 = pymunk.GrooveJoint(spring_anchor, surface, (100, 0), (100, -500), (100, 0))
space.add(groove2)

robot_torso = pymunk.Body(1, pymunk.moment_for_box(1, (50, 50)))
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