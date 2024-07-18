import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np  

# An actual robot actuator with a springy surface on a groove joint

pygame.init()
screen = pygame.display.set_mode((300, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
space = pymunk.Space()
space.gravity = (0, 900)

anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
spring_anchor_shape = pymunk.Circle(anchor, 5)
anchor.position = (150, 500)
space.add(anchor, spring_anchor_shape)

surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
surface.position = (150, 400)
surface_shape = pymunk.Circle(surface, 5)
space.add(surface, surface_shape)

spring1 = pymunk.DampedSpring(anchor, surface, (0, 0), (0, 0), 100, 1000, 0)
space.add(spring1)

surface_groove = pymunk.GrooveJoint(anchor, surface, (0, 0), (0, -500), (0, 0))
space.add(surface_groove)

robot_body = pymunk.Body(2, pymunk.moment_for_circle(1, 0, 25))
robot_body.position = (150, 100)
robot_shape = pymunk.Circle(robot_body, 25)
space.add(robot_body, robot_shape)
robot_body_groove = pymunk.GrooveJoint(anchor, robot_body, (0, 0), (0, -500), (0, 0))
space.add(robot_body_groove)

robot_upper_leg = pymunk.Body(1, pymunk.moment_for_segment(1, (0, 0), (0, 50), 5))
robot_upper_leg.position = (150, 130)
robot_upper_leg_shape = pymunk.Segment(robot_upper_leg, (0, 0), (0, 50), 5)
space.add(robot_upper_leg, robot_upper_leg_shape)
robot_upper_leg_pivot = pymunk.PivotJoint(robot_body, robot_upper_leg, (150, 100))
space.add(robot_upper_leg_pivot)
upper_leg_groove = pymunk.GrooveJoint(anchor, robot_upper_leg, (0, 0), (0, -500), (0, 0))
space.add(upper_leg_groove)

robot_lower_leg = pymunk.Body(1, pymunk.moment_for_segment(1, (0, 0), (0, 50), 5))
robot_lower_leg.position = (150, 200)
robot_lower_leg_shape = pymunk.Segment(robot_lower_leg, (0, 0), (0, 50), 5)
space.add(robot_lower_leg, robot_lower_leg_shape)
robot_lower_leg_pivot = pymunk.PivotJoint(robot_upper_leg, robot_lower_leg, (150, 190))
space.add(robot_lower_leg_pivot)
lower_leg_groove = pymunk.GrooveJoint(anchor, robot_lower_leg, (0, 0), (0, -500), (0, 50))
space.add(lower_leg_groove)


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