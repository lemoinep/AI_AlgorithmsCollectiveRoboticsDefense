# Author(s): Dr. Patrick Lemoine
# Exploring and integrating targeted combat strategies using artificial intelligence.
# This project is currently in the development phase.
# I will gradually incorporate effective solutions as progress is made. 
# Stay tuned!

import pyglet
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import math
import time

# Paramètres simulation
N_ANTS = 30
SPACE_SIZE = 50
ANT_SPEED = 0.3
ATTACK_RANGE = 2.0
TARGET_SPEED = 0.05  # vitesse des cibles mobiles
FIXED_TARGET_INACTIVE_DELAY = 5.0  # secondes avant qu'une cible fixe devienne inactive
MOVING_TARGET_INACTIVE_DELAY = 15.0  # délai plus long pour cibles mobiles
SHRINK_DURATION = 5.0  # durée en secondes pour disparition progressive

# Création fenêtre avec contexte OpenGL 2.1 compatibility profile
config = pyglet.gl.Config(major_version=2, minor_version=1, double_buffer=True)
window = pyglet.window.Window(800, 600, "Simulation Drone 3D - Cibles dynamiques", resizable=True, config=config)
window.set_minimum_size(400, 300)

rot_x, rot_y = 25, -45
zoom = -60

new_targets_spawned = False  # Flag pour éviter multi-créations

class FixedTarget:
    def __init__(self, pos):
        self.pos = list(pos)
        self.active = True
        self.occupied_since = None
        self.inactive_since = None
        self.size = 2.0  # taille initiale

    def update_status(self, ants):
        if not self.active:
            return
        now = time.time()
        occupied = any(
            math.sqrt(sum((a - t)**2 for a, t in zip(ant.pos, self.pos))) < ATTACK_RANGE
            for ant in ants
        )
        if occupied:
            if self.occupied_since is None:
                self.occupied_since = now
            elif now - self.occupied_since >= FIXED_TARGET_INACTIVE_DELAY:
                self.active = False
                self.inactive_since = now
        else:
            self.occupied_since = None

    def update_size_and_lifetime(self):
        if self.active:
            self.size = 2.0
            self.inactive_since = None
            return True
        if self.inactive_since is None:
            self.inactive_since = time.time()
        elapsed = time.time() - self.inactive_since
        ratio = max(0.0, 1.0 - elapsed / SHRINK_DURATION)
        self.size = 2.0 * ratio
        return ratio > 0.01  # True = visible, False = à supprimer

class MovingTarget:
    def __init__(self, pos):
        self.pos = list(pos)
        dir_vec = [random.uniform(-1,1) for _ in range(3)]
        length = math.sqrt(sum(d*d for d in dir_vec))
        self.dir = [d/length for d in dir_vec]
        self.active = True
        self.occupied_since = None
        self.inactive_since = None
        self.size = 2.0

    def update(self):
        if not self.active:
            return  # Arrêt du mouvement si inactive
        for i in range(3):
            self.pos[i] += self.dir[i] * TARGET_SPEED
            if self.pos[i] < -SPACE_SIZE/2 or self.pos[i] > SPACE_SIZE/2:
                self.dir[i] = -self.dir[i]
                self.pos[i] = max(min(self.pos[i], SPACE_SIZE/2), -SPACE_SIZE/2)

    def update_status(self, ants):
        if not self.active:
            return
        now = time.time()
        occupied = any(
            math.sqrt(sum((a - t)**2 for a, t in zip(ant.pos, self.pos))) < ATTACK_RANGE
            for ant in ants
        )
        if occupied:
            if self.occupied_since is None:
                self.occupied_since = now
            elif now - self.occupied_since >= MOVING_TARGET_INACTIVE_DELAY:
                self.active = False
                self.inactive_since = now
        else:
            self.occupied_since = None

    def update_size_and_lifetime(self):
        if self.active:
            self.size = 2.0
            self.inactive_since = None
            return True
        if self.inactive_since is None:
            self.inactive_since = time.time()
        elapsed = time.time() - self.inactive_since
        ratio = max(0.0, 1.0 - elapsed / SHRINK_DURATION)
        self.size = 2.0 * ratio
        return ratio > 0.01

class Ant3D:
    def __init__(self):
        self.pos = [random.uniform(-SPACE_SIZE/2, SPACE_SIZE/2) for _ in range(3)]
        self.state = "exploring"
        self.dir = [random.uniform(-1,1) for _ in range(3)]
        self.normalize_dir()
        self.path = [list(self.pos)]
        self.target = None

    def normalize_dir(self):
        length = math.sqrt(sum(d*d for d in self.dir))
        if length > 0:
            self.dir = [d/length for d in self.dir]
        else:
            self.dir = [1,0,0]

    def choose_target(self, targets):
        active_targets = [t for t in targets if t.active]
        if not active_targets:
            self.target = None
            return
        closest = None
        min_dist = float('inf')
        for t in active_targets:
            pos = t.pos if hasattr(t, 'pos') else t
            dist = math.sqrt(sum((p - q)**2 for p,q in zip(self.pos, pos)))
            if dist < min_dist:
                min_dist = dist
                closest = t
        self.target = closest

    def move_towards(self):
        if self.target is None:
            self.pos = [p + d*ANT_SPEED for p,d in zip(self.pos, self.dir)]
            for i in range(3):
                if self.pos[i] < -SPACE_SIZE/2 or self.pos[i] > SPACE_SIZE/2:
                    self.dir[i] = -self.dir[i]
                    self.pos[i] = max(min(self.pos[i], SPACE_SIZE/2), -SPACE_SIZE/2)
            self.path.append(list(self.pos))
            if len(self.path) > 100:
                self.path.pop(0)
            return

        target_pos = self.target.pos if hasattr(self.target, 'pos') else self.target
        vx = target_pos[0] - self.pos[0]
        vy = target_pos[1] - self.pos[1]
        vz = target_pos[2] - self.pos[2]
        dist = math.sqrt(vx*vx + vy*vy + vz*vz)
        if dist < ATTACK_RANGE:
            self.state = "attacking"
            self.dir = [random.uniform(-1,1) for _ in range(3)]
            self.normalize_dir()
        else:
            self.state = "exploring"
            self.dir = [0.8*d + 0.2*(v/dist) for d,v in zip(self.dir, (vx, vy, vz))]
            self.normalize_dir()
            self.pos = [p + d*ANT_SPEED for p,d in zip(self.pos, self.dir)]
            for i in range(3):
                self.pos[i] = max(-SPACE_SIZE/2, min(SPACE_SIZE/2, self.pos[i]))

        self.path.append(list(self.pos))
        if len(self.path) > 100:
            self.path.pop(0)

    def draw(self):
        if self.state == "exploring":
            glColor3f(1, 0, 0)
        else:
            glColor3f(1, 1, 0)

        glPushMatrix()
        glTranslatef(*self.pos)
        draw_sphere(0.5, 16, 16)
        glPopMatrix()

        glColor3f(0.7, 0.3, 0.3)
        glBegin(GL_LINE_STRIP)
        for p in self.path:
            glVertex3f(*p)
        glEnd()

def draw_sphere(radius=1.0, slices=16, stacks=16):
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)

def draw_axes(length=10):
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)
    glEnd()

def draw_grid(size=50, step=5):
    glColor3f(0.7, 0.7, 0.7)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
    glEnd()

fixed_targets = [
    FixedTarget([15, 0, 15]),
    FixedTarget([-15, 0, 15]),
    FixedTarget([0, 0, 0])
]

moving_targets = [
    MovingTarget([15, 0, -15]),
    MovingTarget([-15, 0, -15])
]

all_targets = fixed_targets + moving_targets

def all_targets_inactive(targets):
    return all(not t.active for t in targets)

ants = [Ant3D() for _ in range(N_ANTS)]

new_targets_spawned = False  # Flag pour éviter multi-créations

@window.event
def on_draw():
    global new_targets_spawned

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(0, 0, zoom)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)

    draw_axes(20)
    draw_grid(25, 5)

    # Mise à jour et nettoyage cibles fixes
    fixed_targets[:] = [t for t in fixed_targets if t.update_size_and_lifetime()]
    # Mise à jour et nettoyage cibles mobiles
    moving_targets[:] = [t for t in moving_targets if t.update_size_and_lifetime()]

    # Dessiner cibles fixes
    for t in fixed_targets:
        glPushMatrix()
        glTranslatef(*t.pos)
        if t.active:
            glColor3f(0, 1, 0)
        else:
            glColor3f(0, 0, 0)
        draw_sphere(t.size, 20, 20)
        glPopMatrix()

    # Dessiner cibles mobiles
    for mt in moving_targets:
        glPushMatrix()
        glTranslatef(*mt.pos)
        if mt.active:
            glColor3f(0, 0.6, 0)
        else:
            glColor3f(0, 0, 0)
        draw_sphere(mt.size, 20, 20)
        glPopMatrix()

    # Mise à jour cibles mobiles (mouvement + statut)
    for mt in moving_targets:
        mt.update()
        mt.update_status(ants)

    # Mise à jour cibles fixes (statut)
    for t in fixed_targets:
        t.update_status(ants)

    # Vérifier si toutes les cibles sont inactives
    if all_targets_inactive(all_targets):
        if not new_targets_spawned:
            n_new_targets = random.randint(1, 3)
            for _ in range(n_new_targets):
                start_z = random.uniform(-3 * SPACE_SIZE, -2 * SPACE_SIZE)
                start_x = random.uniform(-SPACE_SIZE/2, SPACE_SIZE/2)
                start_y = 0
                new_target = MovingTarget([start_x, start_y, start_z])
                moving_targets.append(new_target)
                all_targets.append(new_target)
            new_targets_spawned = True
    else:
        new_targets_spawned = False

    # Répartir les fourmis sur cibles mobiles actives
    assign_ants_to_moving_targets(ants, moving_targets)

    # Drone choisissent une cible si pas assignée ou cible inactive
    for ant in ants:
        if ant.target is None or (ant.target and not ant.target.active):
            ant.choose_target([t for t in all_targets if t.active and t not in moving_targets])
        ant.move_towards()
        ant.draw()

def assign_ants_to_moving_targets(ants, moving_targets):
    active_targets = [t for t in moving_targets if t.active]
    n_targets = len(active_targets)
    if n_targets == 0:
        for ant in ants:
            ant.target = None
        return
    for i, ant in enumerate(ants):
        target_index = i % n_targets
        ant.target = active_targets[target_index]

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height), 0.01, 200.0)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global rot_x, rot_y
    if buttons & pyglet.window.mouse.LEFT:
        rot_x += dy * 0.5
        rot_y += dx * 0.5

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global zoom
    zoom += scroll_y * 2
    zoom = max(-150, min(-10, zoom))

def setup():
    global new_targets_spawned
    new_targets_spawned = False
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.9, 0.9, 0.9, 1)

if __name__ == "__main__":
    setup()
    pyglet.app.run()
