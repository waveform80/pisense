from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Red, Green, Blue
from time import time, sleep
from itertools import cycle, chain
import numpy as np
import io
import rrdtool


def thermometer(offset, reading):
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
    return screen.clip(0, 1)


def hygrometer(offset, reading):
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
    return screen.clip(0, 1)


def barometer(offset, reading):
    p = (max(950, min(1050, reading.pressure)) - 950) / 100 * 64
    screen = array([
        Color('green') if i < int(p) else
        Color('green') * Green(p - int(p)) if i < p else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text(str(round(reading.pressure)),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 8, 0)))
    screen[:text.shape[0], :] += text[:, offset:offset + 8]
    return screen.clip(0, 1)


def bounce(it):
    # bounce('ABC') --> A B C C B A A B C ...
    return cycle(chain(it, reversed(it)))


def switcher(events, readings, database='environ.rrd'):
    rrdtool.create(
        database,            # Filename of the database
        '--no-overwrite',    # Don't overwrite the file if it exists
        '--step', '5s',      # Data will be fed every 5 seconds
        'DS:temperature:GAUGE:1:-70:70',  # Primary store for temperatures
        'DS:humidity:GAUGE:1:0:100',      # Primary store for humidities
        'DS:pressure:GAUGE:1:900:1100',   # Primary store for pressures
        'RRA:AVERAGE:0.5:1:17280',   # Keep 1 day's worth of full-res data
        'RRA:AVERAGE:0.5:60:8064',   # Keep 4 week's worth of 5-minute-res data
        'RRA:AVERAGE:0.5:720:8766',  # Keep 1 year's worth of hour-res data
        'RRA:MIN:0.5:720:8766',      # ... including minimums
        'RRA:MAX:0.5:720:8766',      # ... and maximums
    )
    screens = {
        (thermometer, 'right'): hygrometer,
        (hygrometer, 'left'): thermometer,
        (hygrometer, 'right'): barometer,
        (barometer, 'left'): hygrometer,
    }
    screen = thermometer
    last_update = None
    for event, offset, reading in zip(events, bounce(range(8)), readings):
        anim = 'draw'
        if event is not None and event.pressed:
            try:
                screen = screens[screen, event.direction]
                anim = event.direction
            except KeyError:
                yield array(Color('black')), 'fade'
                break
        now = time()
        if last_update is None or now - last_update > 5:
            last_update = now
            rrdtool.update(
                database,
                'N:{r.temperature}:{r.humidity}:{r.pressure}'.format(r=reading),
            )
        yield screen(offset, reading), anim
        sleep(0.2)


def main():
    with SenseHAT() as hat:
        hat.stick.stream = True
        for a, anim in switcher(hat.stick, hat.environ):
            if anim == 'fade':
                hat.screen.fade_to(a, duration=0.5)
            elif anim == 'right':
                hat.screen.slide_to(a, direction='left', duration=0.5)
            elif anim == 'left':
                hat.screen.slide_to(a, direction='right', duration=0.5)
            else:
                hat.screen.array = a


if __name__ == '__main__':
    main()
