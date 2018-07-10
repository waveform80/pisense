from __future__ import division  # for py2.x compatibility
from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Red
from time import sleep
import numpy as np

def thermometer(reading):
    t = max(0, min(50, reading.temperature)) / 50 * 64
    screen = array([
        Color('red') if i < int(t) else
        Color('red') * Red(t - int(t)) if i < t else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text(str(int(round(reading.temperature))),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 0, 3)))
    screen[:text.shape[0], :text.shape[1]] += text
    return screen.clip(0, 1)

with SenseHAT() as hat:
    for reading in hat.environ:
        hat.screen.array = thermometer(reading)
        sleep(0.5)
