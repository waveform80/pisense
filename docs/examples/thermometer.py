from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Red
from time import sleep
import numpy as np

def thermometer(readings):
    for reading in readings:
        t = max(0, min(50, reading.temperature)) / 50 * 64
        screen = array([
            Color('red') if i < int(t) else
            Color('red') * Red(t - int(t)) if i < t else
            Color('black')
            for i in range(64)
        ])
        screen = np.flipud(screen)
        text = image_to_rgb(draw_text(str(round(reading.temperature)),
                                      'small.pil', foreground=Color('gray')))
        screen[:text.shape[0], :text.shape[1]] += text
        yield screen.clip(0, 1)
        sleep(0.5)

with SenseHAT() as hat:
    for screen in thermometer(hat.environ):
        hat.screen.array = screen
