import pymunk
import numpy as np
import pymunk.pygame_util
import pygame


class Robot:
    def __init__(self, pos_init=(150, 100), length_init=100, stiffness_init=100):
        
        # constants
        self.MASS = m = 1
        self.RADIUS = r = 25
        self.EE_MASS = ee_m = self.MASS*0.7
        self.EE_RADIUS = ee_r = 15
        self.LENGTH_LIMITS = (50,300)
        self.STIFFNESS_LIMITS = (50,400)
        self.ACTUATOR_DIM_MULTIPLIER = adm = 30
        
        inertia = pymunk.moment_for_circle(self.MASS, 0, self.RADIUS)
        self.body = pymunk.Body(self.MASS, inertia)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, self.RADIUS)
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        # the 3 points of the triangle
        p1 = self.body.position
        p2 = (self.body.position[0] + (3*4/5)*adm, self.body.position[1] + (3*3/5)*adm)
        p3 = (self.body.position[0], self.body.position[1] + 5*adm)
        
        upper_arm_dim = (adm*3 - r - ee_r - 10, 10)
        upper_arm_inertia = pymunk.moment_for_box(ee_m*3/5, upper_arm_dim)
        self.upper_arm = pymunk.Body(ee_m*3/5, upper_arm_inertia)
        self.upper_arm_shape = pymunk.Poly.create_box(self.upper_arm, upper_arm_dim)
        self.upper_arm_shape.color = (0, 255, 255, 150)
        self.upper_arm.position = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        self.upper_arm.angle = np.pi*(36.87)/180
        
        lower_arm_dim = (adm*4 - r - ee_r - 10, 10)
        lower_arm_inertia = pymunk.moment_for_box(ee_m*4/5, upper_arm_dim)
        self.lower_arm = pymunk.Body(ee_m*4/5, lower_arm_inertia)
        self.lower_arm_shape = pymunk.Poly.create_box(self.lower_arm, lower_arm_dim)
        self.lower_arm_shape.color = (0, 255, 255, 150)
        self.lower_arm.position = ((p2[0]+p3[0])/2, (p2[1]+p3[1])/2)
        self.lower_arm.angle = np.pi*(90+36.87)/180
        
        self.EE = pymunk.Body(ee_m/10, pymunk.moment_for_circle(ee_m/10, 0, ee_r))
        self.EE_shape = pymunk.Circle(self.EE, ee_r)
        self.EE.position = p3
        
        # self.motor = pymunk.SimpleMotor(self.upper_arm, self.lower_arm, 0)
        self.actuator = pymunk.DampedRotarySpring(self.upper_arm, self.lower_arm, -np.pi/2, 3e5, 0)
        self.shoulder_pin = pymunk.PivotJoint(self.body, self.upper_arm, p1)
        self.elbow_pin = pymunk.PivotJoint(self.upper_arm, self.lower_arm, p2)
        self.wrist_pin = pymunk.PivotJoint(self.lower_arm, self.EE, p3)
        

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    clock = pygame.time.Clock()
    running = True
    
    robot = Robot()
    space = pymunk.Space()
    space.gravity = (0, 900)
    space.add(robot.body, robot.body_shape, 
              robot.upper_arm, robot.upper_arm_shape,
              robot.lower_arm, robot.lower_arm_shape,
              robot.EE, robot.EE_shape,
              robot.actuator, robot.shoulder_pin, robot.elbow_pin, robot.wrist_pin)
    
    anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
    anchor.position = (150,900)
    anchor_shape = pymunk.Circle(anchor, 5)
    
    groove_end = (0, -800)
    robot_body_groove = pymunk.GrooveJoint(anchor, robot.body, (0,0), groove_end, (0,0))
    robot_EE_groove = pymunk.GrooveJoint(anchor, robot.EE, (0,0), groove_end, (0,0))
    space.add(anchor, anchor_shape, robot_body_groove, robot_EE_groove)
    
    surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 15))
    surface.position = (150, 700)
    surface_shape = pymunk.Circle(surface, 15)
    surface_shape.color = (0, 0, 0, 255)
    spring = pymunk.DampedSpring(anchor, surface, (0,0), (0,0), 200, 100, 0)
    surface_groove = pymunk.GrooveJoint(anchor, surface, (0,0), groove_end, (0,0))
    space.add(surface, surface_shape, surface_groove, spring)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))
        
        space.step(1/60)
        space.debug_draw(pymunk.pygame_util.DrawOptions(screen))
        
        print(robot.upper_arm_shape.get_vertices())
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()