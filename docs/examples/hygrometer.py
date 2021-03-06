from __future__ import division  # for py2.x compatibility
from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Blue
from time import sleep
import numpy as np

def hygrometer(reading):
    h = reading.humidity / 100 * 64
    screen = array([
        Color('#008') if i < int(h) else
        Color('#008') * Blue(h - int(h)) if i < h else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text('^^' if reading.humidity > 99 else
                                  str(int(round(reading.humidity))),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 0, 3)))
    screen[:text.shape[0], :text.shape[1]] += text
    return screen.clip(0, 1)

with SenseHAT() as hat:
    for reading in hat.environ:
        hat.screen.array = hygrometer(reading)
        sleep(0.5)
