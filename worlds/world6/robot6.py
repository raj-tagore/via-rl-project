import pymunk
import numpy as np
import pymunk.pygame_util
import math


class Robot:
    def __init__(self, pos_init=(150, 150), angle_init=np.pi/2, stiffness_init=8e5):
        
        # constants
        self.MASS = m = 1
        self.RADIUS = r = 30
        self.EE_MASS = ee_m = self.MASS*0.2
        self.EE_RADIUS = ee_r = 15 
        self.ANGLE_LIMITS = (np.pi/6, np.pi)
        self.STIFFNESS_LIMITS = (1e4,5e6)
        self.ACTUATOR_DIM_MULTIPLIER = adm = 40 
        
        # Robot's pymunk Body
        # position and velocity can be accessed by this body instance
        inertia = pymunk.moment_for_circle(self.MASS, 0, self.RADIUS)
        self.body = pymunk.Body(self.MASS, inertia)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, self.RADIUS)
        self.body_shape.filter = pymunk.ShapeFilter(group=1)
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        # the 3 points of the triangle
        self.p1 = p1 = self.body.position
        self.p2 = p2 = (self.body.position[0] - (3*4/5)*adm, self.body.position[1] + (3*3/5)*adm)
        self.p3 = p3 = (self.body.position[0], self.body.position[1] + 5*adm)
        
        self.upper_arm_init_angle = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        self.lower_arm_init_angle = math.atan2(p3[1]-p2[1], p3[0]-p2[0])
        
        upper_arm_dim = (adm*3, 10)
        upper_arm_inertia = pymunk.moment_for_box(ee_m*3/5, upper_arm_dim)
        self.upper_arm = pymunk.Body(ee_m*3/5, upper_arm_inertia)
        self.upper_arm_shape = pymunk.Poly.create_box(self.upper_arm, upper_arm_dim)
        self.upper_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.upper_arm_shape.color = (0, 255, 255, 150)
        self.upper_arm.position = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        self.upper_arm.angle = self.upper_arm_init_angle
        
        lower_arm_dim = (adm*4, 10)
        lower_arm_inertia = pymunk.moment_for_box(ee_m*4/5, upper_arm_dim)
        self.lower_arm = pymunk.Body(ee_m*4/5, lower_arm_inertia)
        self.lower_arm_shape = pymunk.Poly.create_box(self.lower_arm, lower_arm_dim)
        self.lower_arm_shape.filter = pymunk.ShapeFilter(group=1)
        self.lower_arm_shape.color = (0, 255, 255, 150)
        self.lower_arm.position = ((p2[0]+p3[0])/2, (p2[1]+p3[1])/2)
        self.lower_arm.angle = self.lower_arm_init_angle
        
        self.EE = pymunk.Body(ee_m/10, pymunk.moment_for_circle(ee_m/10, 0, ee_r))
        self.EE_shape = pymunk.Circle(self.EE, ee_r)
        self.EE_shape.filter = pymunk.ShapeFilter(group=1)
        self.EE.position = p3
        self.EE_contact = 0
        self.EE_previous_velocity = 0
        self.EE_acceleration = 0
        
        self.actuator = pymunk.DampedRotarySpring(self.upper_arm, self.lower_arm, angle_init, stiffness_init, 0)
        self.shoulder_pin = pymunk.PivotJoint(self.body, self.upper_arm, p1)
        self.elbow_pin = pymunk.PivotJoint(self.upper_arm, self.lower_arm, p2)
        self.wrist_pin = pymunk.PivotJoint(self.lower_arm, self.EE, p3)
        
    def do_action(self, action, noise=False, theta_control='direct', k_control='direct'):
        action = np.array(action)
        
        alpha = 30
        # constants
        theta_min = self.ANGLE_LIMITS[0]
        theta_max = self.ANGLE_LIMITS[1]
        theta_increment_multiplier = (theta_max-theta_min)/alpha
        k_min = self.STIFFNESS_LIMITS[0]
        k_max = self.STIFFNESS_LIMITS[1]
        k_increment_multiplier = (k_max-k_min)/alpha
        
        # add noise to the action
        if noise:
            action += np.random.normal(0, noise, action.shape)
        
        # controlling angle    
        if theta_control == 'direct':
            angle = action[0]*(theta_max - theta_min)/2 + (theta_max + theta_min)/2
        elif theta_control == 'incremental':
            angle = self.actuator.rest_angle + theta_increment_multiplier*action[0]
            angle = np.clip(angle, theta_min, theta_max)
            
        # controlling stiffness
        if k_control == 'direct':
            stiffness = action[1]*(k_max - k_min)/2 + (k_max + k_min)/2
        elif k_control == 'incremental':
            stiffness = self.actuator.stiffness + k_increment_multiplier*action[1]
            stiffness = np.clip(stiffness, k_min, k_max)
        
        self.actuator.rest_angle = angle
        self.actuator.stiffness = stiffness
        
    def reset(self, mass=1, pos_init=(150, 150)):
        
        self.body.mass = mass
        self.body.position = pos_init
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        ee_m = self.MASS*0.2
        self.upper_arm.mass = ee_m*3/5
        self.lower_arm.mass = ee_m*4/5
        self.EE.mass = ee_m/10
        
        adm = self.ACTUATOR_DIM_MULTIPLIER
        p1 = self.body.position
        p2 = (self.body.position[0] - (3*4/5)*adm, self.body.position[1] + (3*3/5)*adm)
        p3 = (self.body.position[0], self.body.position[1] + 5*adm)
        
        self.upper_arm.position = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        self.upper_arm.angle = self.upper_arm_init_angle
        self.lower_arm.position = ((p2[0]+p3[0])/2, (p2[1]+p3[1])/2)
        self.lower_arm.angle = self.lower_arm_init_angle
        self.EE.position = p3
        
        self.actuator.rest_angle = np.pi/2
        self.actuator.stiffness = 8e5
        
    def update_accelerations(self):
        
        body_velocity_change = self.body.velocity.y - self.body_previous_velocity
        self.body_acceleration = body_velocity_change
        
        EE_velocity_change = self.EE.velocity.y - self.EE_previous_velocity
        self.EE_acceleration = EE_velocity_change
        
        self.body_previous_velocity = self.body.velocity.y
        self.EE_previous_velocity = self.EE.velocity.y
        
        
            
        

        
        
    