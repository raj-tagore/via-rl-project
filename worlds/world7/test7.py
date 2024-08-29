import pymunk
import numpy as np
import pymunk.pygame_util
import math
import pygame

class Robot:
    def __init__(self):
        
        # constants
        self.MASS = m = 1
        self.LENGTH = 300
        self.EE_MASS = ee_m = self.MASS*0.2
        
        self.ANGLE_LIMITS = (np.pi/6, np.pi)
        self.STIFFNESS_LIMITS = (100,3e6)
        self.ACTUATOR_DIM_MULTIPLIER = adm = 40
        
        # Robot's pymunk Body
        inertia = pymunk.moment_for_box(m, (self.LENGTH+100, 40))
        self.body = pymunk.Body(m, inertia)
        self.body.position = (500, 150)
        self.body_shape = pymunk.Poly.create_box(self.body, (self.LENGTH+100, 40))
        self.body_shape.filter = pymunk.ShapeFilter(group=1)
        self.body_shape.color = (0, 0, 0, 150)
        self.body_previous_velocity = 0 
        self.body_acceleration = 0
        
        # the 3 points of the triangle
        self.left_p1 = lp1 = (self.body.position[0] - self.LENGTH/2, self.body.position[1])
        self.left_p2 = lp2 = (lp1[0] - (3*4/5)*adm, lp1[1] + (3*3/5)*adm)
        self.left_p3 = lp3 = (lp1[0], lp1[1] + 5*adm)
        
        self.right_p1 = rp1 = (self.body.position[0] + self.LENGTH/2, self.body.position[1])
        self.right_p2 = rp2 = (rp1[0] - (3*4/5)*adm, rp1[1] + (3*3/5)*adm)
        self.right_p3 = rp3 = (rp1[0], rp1[1] + 5*adm)
        
        self.upper_arm_init_angle = math.atan2(lp2[1]-lp1[1], lp2[0]-lp1[0])
        self.lower_arm_init_angle = math.atan2(lp3[1]-lp2[1], lp3[0]-lp2[0])
        
        upper_arm_dim = (adm*3, 10)
        upper_arm_inertia = pymunk.moment_for_box(ee_m*3/5, upper_arm_dim)
        
        self.left_upper_arm = pymunk.Body(ee_m*3/5, upper_arm_inertia)
        self.left_upper_arm_shape = pymunk.Poly.create_box(self.left_upper_arm, upper_arm_dim)
        self.left_upper_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.left_upper_arm_shape.color = (0, 255, 255, 150)
        self.left_upper_arm.position = ((lp1[0]+lp2[0])/2, (lp1[1]+lp2[1])/2)
        self.left_upper_arm.angle = self.upper_arm_init_angle
        
        self.right_upper_arm = pymunk.Body(ee_m*3/5, upper_arm_inertia)
        self.right_upper_arm_shape = pymunk.Poly.create_box(self.right_upper_arm, upper_arm_dim)
        self.right_upper_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.right_upper_arm_shape.color = (0, 255, 255, 150)
        self.right_upper_arm.position = ((rp1[0]+rp2[0])/2, (rp1[1]+rp2[1])/2)
        self.right_upper_arm.angle = self.upper_arm_init_angle
        
        lower_arm_dim = (adm*4, 10)
        lower_arm_inertia = pymunk.moment_for_box(ee_m*4/5, lower_arm_dim)
        
        self.left_lower_arm = pymunk.Body(ee_m*4/5, lower_arm_inertia)
        self.left_lower_arm_shape = pymunk.Poly.create_box(self.left_lower_arm, lower_arm_dim)
        self.left_lower_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.left_lower_arm_shape.color = (0, 255, 255, 150)
        self.left_lower_arm.position = ((lp2[0]+lp3[0])/2, (lp2[1]+lp3[1])/2)
        self.left_lower_arm.angle = self.lower_arm_init_angle
        
        self.right_lower_arm = pymunk.Body(ee_m*4/5, lower_arm_inertia)
        self.right_lower_arm_shape = pymunk.Poly.create_box(self.right_lower_arm, lower_arm_dim)
        self.right_lower_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.right_lower_arm_shape.color = (0, 255, 255, 150)
        self.right_lower_arm.position = ((rp2[0]+rp3[0])/2, (rp2[1]+rp3[1])/2)
        self.right_lower_arm.angle = self.lower_arm_init_angle
        
        self.left_EE = pymunk.Body(ee_m/10, pymunk.moment_for_circle(ee_m/10, 0, 15))
        self.left_EE_shape = pymunk.Circle(self.left_EE, 15)
        self.left_EE_shape.filter = pymunk.ShapeFilter(group=1)
        self.left_EE.position = lp3
        self.left_EE_contact = 0
        self.left_EE_previous_velocity = 0
        self.left_EE_acceleration = 0
        
        self.right_EE = pymunk.Body(ee_m/10, pymunk.moment_for_circle(ee_m/10, 0, 15))
        self.right_EE_shape = pymunk.Circle(self.right_EE, 15)
        self.right_EE_shape.filter = pymunk.ShapeFilter(group=1)
        self.right_EE.position = rp3
        self.right_EE_contact = 0
        self.right_EE_previous_velocity = 0
        self.right_EE_acceleration = 0
        
        angle_init = np.pi/2
        stiffness_init = 8e5
        upper_arm_angle_limits = uaal = (2*np.pi/3, np.pi)
        
        self.left_actuator = pymunk.DampedRotarySpring(self.left_upper_arm, self.left_lower_arm, angle_init, stiffness_init, 0)
        self.left_shoulder_pin = pymunk.PivotJoint(self.body, self.left_upper_arm, lp1) 
        self.left_shoulder_rotary_limit = pymunk.RotaryLimitJoint(self.body, self.left_upper_arm, uaal[0], uaal[1])
        self.left_elbow_pin = pymunk.PivotJoint(self.left_upper_arm, self.left_lower_arm, lp2)
        self.left_wrist_pin = pymunk.PivotJoint(self.left_lower_arm, self.left_EE, lp3)
        
        self.right_actuator = pymunk.DampedRotarySpring(self.right_upper_arm, self.right_lower_arm, angle_init, stiffness_init, 0)
        self.right_shoulder_pin = pymunk.PivotJoint(self.body, self.right_upper_arm, rp1)
        self.right_shoulder_rotary_limit = pymunk.RotaryLimitJoint(self.body, self.right_upper_arm, uaal[0], uaal[1])
        self.right_elbow_pin = pymunk.PivotJoint(self.right_upper_arm, self.right_lower_arm, rp2)
        self.right_wrist_pin = pymunk.PivotJoint(self.right_lower_arm, self.right_EE, rp3)
        
        

        
        
class Env:
    def __init__(self):
        
        # constants
        self.REST_LENGTH_LIMITS = (100, 400)
        self.STIFFNESS_LIMITS = (20, 800)
        self.SURFACE_MASS = 2
        self.SURFACE_RADIUS = 15
        
        self.left_anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.left_anchor_shape = pymunk.Circle(self.left_anchor, 5)
        self.left_anchor.position = (150, 900)
        
        self.right_anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.right_anchor_shape = pymunk.Circle(self.right_anchor, 5)
        self.right_anchor.position = (850, 900)
        
        inertia = pymunk.moment_for_box(self.SURFACE_MASS, (1000, 20))
        self.surface = pymunk.Body(self.SURFACE_MASS, inertia)
        self.surface_shape = pymunk.Poly.create_box(self.surface, (1000, 20))
        self.surface.position = (500, 450)
        
        self.left_spring = pymunk.DampedSpring(self.left_anchor, self.surface, (0, 0), (-350, 0), 300, 100, 0)
        self.right_spring = pymunk.DampedSpring(self.right_anchor, self.surface, (0, 0), (350, 0), 300, 100, 0)
        

def restrict_to_1d(*args):
    for obj in args:
        obj.velocity = (0,obj.velocity[1])
        
        
        
               
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    clock = pygame.time.Clock()
    running = True
    
    robot = Robot()
    env = Env() 
    space = pymunk.Space()
    space.gravity = (0, 2000)
    space.add(robot.body, robot.body_shape, 
              robot.left_upper_arm, robot.left_upper_arm_shape,
              robot.left_lower_arm, robot.left_lower_arm_shape,
              robot.left_EE, robot.left_EE_shape,
              robot.left_actuator, 
              robot.left_shoulder_pin, robot.left_shoulder_rotary_limit,
              robot.left_elbow_pin, robot.left_wrist_pin,
              robot.right_upper_arm, robot.right_upper_arm_shape,
              robot.right_lower_arm, robot.right_lower_arm_shape,
              robot.right_EE, robot.right_EE_shape,
              robot.right_actuator, 
              robot.right_shoulder_pin, robot.right_shoulder_rotary_limit,
              robot.right_elbow_pin, robot.right_wrist_pin)
    
    space.add(env.left_anchor, env.left_anchor_shape, 
              env.right_anchor, env.right_anchor_shape, 
              env.surface, env.surface_shape,
              env.left_spring, env.right_spring)
    
    
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255,255,255))
        
        restrict_to_1d(robot.body, env.surface)
        space.step(1/120)
        
        space.debug_draw(pymunk.pygame_util.DrawOptions(screen))
        
        pygame.display.flip()
        clock.tick(60)
    
    
    
    
        
        
        
        