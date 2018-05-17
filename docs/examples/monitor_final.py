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


def create_database(database):
    try:
        rrdtool.create(
            database,            # Filename of the database
            '--no-overwrite',    # Don't overwrite the file if it exists
            '--step', '5s',      # Data will be fed at least every 5 seconds
            'DS:temperature:GAUGE:1m:-70:70',  # Primary store for temperatures
            'DS:humidity:GAUGE:1m:0:100',      # Primary store for humidities
            'DS:pressure:GAUGE:1m:900:1100',   # Primary store for pressures
            'RRA:AVERAGE:0.5:5s:1d',  # Keep 1 day's worth of full-res data
            'RRA:AVERAGE:0.5:5m:1M',  # Keep 1 month of 5-minute-res data
            'RRA:AVERAGE:0.5:1h:1y',  # Keep 1 year of hourly data
            'RRA:MIN:0.5:1h:1y',      # ... including minimums
            'RRA:MAX:0.5:1h:1y',      # ... and maximums
            'RRA:AVERAGE:0.5:1d:10y', # Keep 10 years of daily data
            'RRA:MIN:0.5:1d:10y',     # ... including minimums
            'RRA:MAX:0.5:1d:10y',     # ... and maximums
        )
    except rrdtool.OperationalError:
        pass # file exists; ignore the error


def update_database(database, reading):
    data = 'N:{r.temperature}:{r.humidity}:{r.pressure}'.format(r=reading)
    rrdtool.update(database, data)


def switcher(events, readings, database='environ.rrd'):
    create_database(database)
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
            # Only update the database every 5 seconds
            last_update = now
            update_database(database, reading)
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
