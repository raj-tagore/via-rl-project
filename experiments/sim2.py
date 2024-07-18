import pymunk
import pymunk.pygame_util
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

space = pymunk.Space()
space.gravity = (0, 90)

base_body = pymunk.Body(body_type=pymunk.Body.STATIC)
base_body.position = (300, 300)
base_shape = pymunk.Poly.create_box(base_body, (50, 50))
space.add(base_body, base_shape)

arm_body = pymunk.Body(1, pymunk.moment_for_box(1, (20, 100)))
arm_body.position = (300, 200)
arm_shape = pymunk.Poly.create_box(arm_body, (20, 100))
space.add(arm_body, arm_shape)

pivot_joint = pymunk.PivotJoint(base_body, arm_body, (300, 300))
space.add(pivot_joint)

motor_joint = pymunk.SimpleMotor(base_body, arm_body, 10)
space.add(motor_joint)

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