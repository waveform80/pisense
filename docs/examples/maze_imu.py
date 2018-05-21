import numpy as np
import pisense as ps
from random import sample
from colorzero import Color
from time import sleep


def main():
    width = height = 8
    colors = {
        'unvisited': Color('black'),
        'visited':   Color('green'),
        'wall':      Color('white'),
        'ball':      Color('red'),
        'goal':      Color('yellow'),
    }
    with ps.SenseHAT() as hat:
        maze = generate_maze(width, height, colors)
        inputs = moves(hat.imu)
        outputs = game(maze, colors, inputs)
        display(hat.screen, outputs)


def moves(imu):
    for reading in imu:
        delta_x = round(max(-1, min(1, reading.accel.x)))
        delta_y = round(max(-1, min(1, reading.accel.y)))
        if delta_x != 0 or delta_y != 0:
            yield delta_y, delta_x
        sleep(1/10)


def display(screen, states):
    for anim, data in states:
        if anim == 'fade':
            screen.fade_to(data)
        elif anim == 'zoom':
            screen.zoom_to(data)
        elif anim == 'show':
            screen.array = data
        elif anim == 'scroll':
            screen.scroll_text(data, background=Color('red'))
        else:
            assert False


def game(maze, colors, moves):
    height, width = maze.shape
    y, x = (1, 1)
    maze[y, x] = colors['ball']
    left, right = clamp(x, width)
    top, bottom = clamp(y, height)
    yield 'fade', maze[top:bottom, left:right]
    for delta_y, delta_x in moves:
        if Color(*maze[y + delta_y, x + delta_x]) != colors['wall']:
            maze[y, x] = colors['visited']
            y += delta_y
            x += delta_x
            if Color(*maze[y, x]) == colors['goal']:
                yield from winners_cup()
                break
            else:
                maze[y, x] = colors['ball']
                left, right = clamp(x, width)
                top, bottom = clamp(y, height)
                yield 'show', maze[top:bottom, left:right]
    yield 'fade', ps.array(Color('black'))


def generate_maze(width, height, colors):
    walls = generate_walls(width, height)
    maze = ps.array(shape=(2 * height + 1, 2 * width + 1))
    maze[...] = colors['unvisited']
    maze[::2, ::2] = colors['wall']
    for a, b in walls:
        ay, ax = a
        by, bx = b
        y = 2 * by + 1
        x = 2 * bx + 1
        if ay == by:
            maze[y, x - 1] = colors['wall']
        else:
            maze[y - 1, x] = colors['wall']
    maze[0, :] = maze[:, 0] = colors['wall']
    maze[-1, :] = maze[:, -1] = colors['wall']
    maze[-2, -2] = colors['goal']
    return maze


def generate_walls(width, height):
    # Generate the maze with Kruskal's algorithm (there's better choices,
    # but this is a simple demo!)
    sets = {frozenset({(y, x)}) for y in range(height) for x in range(width)}
    walls = set()
    for y in range(height):
        for x in range(width):
            if x > 0:
                # Add west wall
                walls.add(((y, x - 1), (y, x)))
            if y > 0:
                # Add north wall
                walls.add(((y - 1, x), (y, x)))
    for wall in sample(list(walls), k=len(walls)):
        # For a random wall, find the sets containing the adjacent cells
        a, b = wall
        set_a = set_b = None
        for s in sets:
            if {a, b} <= s:
                set_a = set_b = s
            elif a in s:
                set_a = s
            elif b in s:
                set_b = s
            if set_a is not None and set_b is not None:
                break
        # If the sets aren't the same, the cells aren't reachable; remove the
        # wall between them
        if set_a is not set_b:
            sets.add(set_a | set_b)
            sets.remove(set_a)
            sets.remove(set_b)
            walls.remove(wall)
        if len(sets) == 1:
            break
    assert len(sets) == 1
    assert sets.pop() == {(y, x) for y in range(height) for x in range(width)}
    return walls


def clamp(pos, limit, window=8):
    low, high = pos - window // 2, pos + window // 2
    if low < 0:
        high += -low
        low = 0
    elif high > limit:
        low -= high - limit
        high = limit
    return low, high


def winners_cup():
    r = Color('red')
    y = Color('yellow')
    W = Color('white')
    yield 'zoom', ps.array([
        r, r, W, y, y, y, r, r,
        r, r, W, y, y, y, r, r,
        r, r, W, y, y, y, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, W, y, y, y, r, r,
    ])
    sleep(2)
    yield 'fade', ps.array(r)
    yield 'scroll', 'You win!'


if __name__ == '__main__':
    main()
