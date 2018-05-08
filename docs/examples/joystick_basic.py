from pisense import SenseHAT, array
from colorzero import Color

def movements(hat):
    for event in hat.stick:
        if event.pressed:
            try:
                yield {
                    'left':  (-1, 0),
                    'right': (1, 0),
                    'up':    (0, -1),
                    'down':  (0, 1),
                }[event.direction]
            except KeyError:
                break  # enter exits

def arrays(moves):
    a = array(Color('black'))  # blank screen
    x = y = 3
    a[y, x] = Color('white')
    yield a  # initial position
    for dx, dy in moves:
        a[y, x] = Color('black')
        x = min(7, max(0, x + dx))
        y = min(7, max(0, y + dy))
        a[y, x] = Color('white')
        yield a
    a[y, x] = Color('black')
    yield a  # exit with blank display

with SenseHAT() as hat:
    for a in arrays(movements(hat)):
        hat.screen.array = a
