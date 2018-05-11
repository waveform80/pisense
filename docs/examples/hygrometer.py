from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Blue
from time import sleep
import numpy as np

def hygrometer(readings):
    for reading in readings:
        h = reading.humidity / 100 * 64
        screen = array([
            Color('#008') if i < int(h) else
            Color('#008') * Blue(h - int(h)) if i < h else
            Color('black')
            for i in range(64)
        ])
        screen = np.flipud(screen)
        text = image_to_rgb(draw_text('^^' if reading.humidity > 99 else
                                      str(round(reading.humidity)),
                                      'small.pil', foreground=Color('gray')))
        screen[:text.shape[0], :text.shape[1]] += text
        yield screen.clip(0, 1)
        sleep(0.5)

with SenseHAT() as hat:
    for screen in hygrometer(hat.environ):
        hat.screen.array = screen
