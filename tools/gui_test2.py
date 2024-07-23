import pygame
import pygame_gui
import pymunk
import pymunk.pygame_util

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

space = pymunk.Space()
space.gravity = (0, 1000)
anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
anchor.position = 400, 300
anchor_shape = pymunk.Circle(anchor, 5)
space.add(anchor, anchor_shape)
body = pymunk.Body(1, 100)
body.position = 400, 100
shape = pymunk.Circle(body, 20)
space.add(body, shape)
spring = pymunk.DampedSpring(anchor, body, (0, 0), (0, 0), 100, 100, 0)
space.add(spring)

# Set up the window
window_size = (800, 600)
window_surface = pygame.display.set_mode(window_size)
pygame.display.set_caption('Pygame GUI Slider Example')
draw_options = pymunk.pygame_util.DrawOptions(window_surface)

# Create a UI Manager
ui_manager = pygame_gui.UIManager(window_size)

# Create a slider
slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((350, 275), (100, 50)),
    start_value=50,
    value_range=(0, 100),
    manager=ui_manager
)

# Create a label to display the slider value
slider_value_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((350, 325), (100, 50)),
    text=f'Slider Value: {slider.get_current_value()}',
    manager=ui_manager
)

# Main loop
clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick(60) / 1000.0  # Limit frame rate to 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
            
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == slider:
                spring.rest_length = 2*event.value
                slider_value_label.set_text(f'Slider Value: {slider.get_current_value()}')

        # Pass events to the UI Manager
        ui_manager.process_events(event)

    # Update the UI Manager
    ui_manager.update(time_delta)

    # Clear the screen
    window_surface.fill((0, 0, 0))
    
    space.step(1/60.0)

    # Draw the UI
    ui_manager.draw_ui(window_surface)
    
    space.debug_draw(draw_options)

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
