# Author(s): Dr. Patrick Lemoine

# Simulation: Collective Robotics Defense with AI
# - Drones attack targets.
# - Targets can destroy drones if attacked for a certain duration.
# - Statistics are displayed in a column in the top-right corner.


import pyglet
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import math
import time

# === Simulation Parameters ===
N_DA3DRS = 100
SPACE_SIZE = 50
DA3DR_SPEED = 0.3
ATTACK_RANGE = 2.0
TARGET_SPEED = 0.05
FIXED_TARGET_INACTIVE_DELAY = 5.0
MOVING_TARGET_INACTIVE_DELAY = 15.0
SHRINK_DURATION = 5.0
nbRandomTarget = 5
DRONE_DESTRUCTION_DELAY = 4.0   # seconds a drone must attack a target to be destroyed
DRONE_RECOVERY_DELAY = 10.0     # seconds before a destroyed drone is reactivated

config = pyglet.gl.Config(major_version=2, minor_version=1, double_buffer=True)
window = pyglet.window.Window(800, 600, "Collective Robotics Defense - Dynamic Targets", resizable=True, config=config)
window.set_minimum_size(400, 300)

rot_x, rot_y = 25, -45
zoom = -60
new_targets_spawned = False

# === Statistics Tracking ===
stats = {
    "active_drones": N_DA3DRS,
    "destroyed_drones": 0,
    "active_targets": 0,
    "destroyed_targets": 0
}

# === Drone Class ===
class DA3DR3D:
    """
    Drone agent. Can be destroyed by targets after a certain attack duration.
    After a cooldown, destroyed drones are reactivated at a random position.
    """
    def __init__(self):
        self.pos = [random.uniform(-SPACE_SIZE/2, SPACE_SIZE/2) for _ in range(3)]
        self.state = "exploring"
        self.dir = [random.uniform(-1,1) for _ in range(3)]
        self.normalize_dir()
        self.path = [list(self.pos)]
        self.target = None
        self.active = True
        self.destroyed_time = None
        self.attacking_target = None
        self.attack_start_time = None

    def normalize_dir(self):
        length = math.sqrt(sum(d*d for d in self.dir))
        if length > 0:
            self.dir = [d/length for d in self.dir]
        else:
            self.dir = [1,0,0]

    def destroy(self):
        """Deactivate the drone and set destruction time."""
        if self.active:
            self.active = False
            self.destroyed_time = time.time()
            stats["active_drones"] -= 1
            stats["destroyed_drones"] += 1

    def try_recover(self, cooldown=DRONE_RECOVERY_DELAY):
        """Reactivate the drone after cooldown."""
        if not self.active and self.destroyed_time is not None:
            if time.time() - self.destroyed_time > cooldown:
                self.active = True
                self.destroyed_time = None
                # Respawn at a random position and reset state
                self.pos = [random.uniform(-SPACE_SIZE/2, SPACE_SIZE/2) for _ in range(3)]
                self.dir = [random.uniform(-1,1) for _ in range(3)]
                self.normalize_dir()
                self.path = [list(self.pos)]
                self.state = "exploring"
                self.target = None
                self.attacking_target = None
                self.attack_start_time = None
                stats["active_drones"] += 1
                stats["destroyed_drones"] -= 1

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
        if not self.active:
            return  # Destroyed drones do not move
        if self.target is None:
            self.pos = [p + d*DA3DR_SPEED for p,d in zip(self.pos, self.dir)]
            for i in range(3):
                if self.pos[i] < -SPACE_SIZE/2 or self.pos[i] > SPACE_SIZE/2:
                    self.dir[i] = -self.dir[i]
                    self.pos[i] = max(min(self.pos[i], SPACE_SIZE/2), -SPACE_SIZE/2)
            self.path.append(list(self.pos))
            if len(self.path) > 100:
                self.path.pop(0)
            self.attacking_target = None
            self.attack_start_time = None
            return

        target_pos = self.target.pos if hasattr(self.target, 'pos') else self.target
        vx = target_pos[0] - self.pos[0]
        vy = target_pos[1] - self.pos[1]
        vz = target_pos[2] - self.pos[2]
        dist = math.sqrt(vx*vx + vy*vy + vz*vz)

        if dist < ATTACK_RANGE:
            self.state = "attacking"
            if self.attacking_target == self.target:
                pass
            else:
                self.attacking_target = self.target
                self.attack_start_time = time.time()
            self.dir = [random.uniform(-1,1) for _ in range(3)]
            self.normalize_dir()
        else:
            self.state = "exploring"
            self.dir = [0.8*d + 0.2*(v/dist) for d,v in zip(self.dir, (vx, vy, vz))]
            self.normalize_dir()
            self.attacking_target = None
            self.attack_start_time = None

        self.pos = [p + d*DA3DR_SPEED for p,d in zip(self.pos, self.dir)]
        for i in range(3):
            self.pos[i] = max(-SPACE_SIZE/2, min(SPACE_SIZE/2, self.pos[i]))
        self.path.append(list(self.pos))
        if len(self.path) > 100:
            self.path.pop(0)

    def draw(self):
        if not self.active:
            return  # Do not draw destroyed drones
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

# === Target Classes ===
class FixedTarget:
    """
    Stationary target. Can destroy attacking drones after a certain duration.
    """
    def __init__(self, pos):
        self.pos = list(pos)
        self.active = True
        self.occupied_since = None
        self.inactive_since = None
        self.size = 2.0
        self.attacking_drones = {}  # drone: attack_start_time

    def update_status(self, DA3DRs):
        if not self.active:
            return
        now = time.time()
        occupied = False
        for drone in DA3DRs:
            if not drone.active:
                continue
            dist = math.sqrt(sum((a - t)**2 for a, t in zip(drone.pos, self.pos)))
            if dist < ATTACK_RANGE:
                occupied = True
                if drone not in self.attacking_drones:
                    self.attacking_drones[drone] = now
                elif now - self.attacking_drones[drone] >= DRONE_DESTRUCTION_DELAY:
                    drone.destroy()
                    del self.attacking_drones[drone]
            else:
                if drone in self.attacking_drones:
                    del self.attacking_drones[drone]
        if occupied:
            if self.occupied_since is None:
                self.occupied_since = now
            elif now - self.occupied_since >= FIXED_TARGET_INACTIVE_DELAY:
                self.active = False
                self.inactive_since = now
                stats["active_targets"] -= 1
                stats["destroyed_targets"] += 1
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

class MovingTarget:
    """
    Mobile target. Can destroy attacking drones after a certain duration.
    """
    def __init__(self, pos):
        self.pos = list(pos)
        dir_vec = [random.uniform(-1,1) for _ in range(3)]
        length = math.sqrt(sum(d*d for d in dir_vec))
        self.dir = [d/length for d in dir_vec]
        self.active = True
        self.occupied_since = None
        self.inactive_since = None
        self.size = 2.0
        self.attacking_drones = {}  # drone: attack_start_time

    def update(self):
        if not self.active:
            return
        for i in range(3):
            self.pos[i] += self.dir[i] * TARGET_SPEED
            if self.pos[i] < -SPACE_SIZE/2 or self.pos[i] > SPACE_SIZE/2:
                self.dir[i] = -self.dir[i]
                self.pos[i] = max(min(self.pos[i], SPACE_SIZE/2), -SPACE_SIZE/2)

    def update_status(self, DA3DRs):
        if not self.active:
            return
        now = time.time()
        occupied = False
        for drone in DA3DRs:
            if not drone.active:
                continue
            dist = math.sqrt(sum((a - t)**2 for a, t in zip(drone.pos, self.pos)))
            if dist < ATTACK_RANGE:
                occupied = True
                if drone not in self.attacking_drones:
                    self.attacking_drones[drone] = now
                elif now - self.attacking_drones[drone] >= DRONE_DESTRUCTION_DELAY:
                    drone.destroy()
                    del self.attacking_drones[drone]
            else:
                if drone in self.attacking_drones:
                    del self.attacking_drones[drone]
        if occupied:
            if self.occupied_since is None:
                self.occupied_since = now
            elif now - self.occupied_since >= MOVING_TARGET_INACTIVE_DELAY:
                self.active = False
                self.inactive_since = now
                stats["active_targets"] -= 1
                stats["destroyed_targets"] += 1
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

# === Visualization Utilities ===
def draw_sphere(radius=1.0, slices=16, stacks=16):
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)

def draw_axes(length=10):
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
    glEnd()

def draw_grid(size=50, step=5):
    glColor3f(0.7, 0.7, 0.7)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, 0, -size); glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i); glVertex3f(size, 0, i)
    glEnd()

def draw_stats():
    """
    Draws simulation statistics in a column in the top-right corner.
    """
    stats_lines = [
        f"Active Drones:     {stats['active_drones']}",
        f"Destroyed Drones:  {stats['destroyed_drones']}",
        f"Active Targets:    {stats['active_targets']}",
        f"Destroyed Targets: {stats['destroyed_targets']}"
    ]
    y = window.height - 10
    glDisable(GL_DEPTH_TEST)
    for line in stats_lines:
        label = pyglet.text.Label(
            line,
            font_name='Arial',
            font_size=14,
            x=window.width - 10, y=y,
            anchor_x='right', anchor_y='top',
            color=(0, 0, 0, 255)
        )
        label.draw()
        y -= 22  # Move down for next line
    glEnable(GL_DEPTH_TEST)

# === Entity Initialization ===
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
stats["active_targets"] = len(all_targets)
DA3DRs = [DA3DR3D() for _ in range(N_DA3DRS)]
stats["active_drones"] = N_DA3DRS
new_targets_spawned = False

def all_targets_inactive(targets):
    return all(not t.active for t in targets)

# === Main Simulation Loop ===
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

    # Try to recover destroyed drones (simulate repair/recharge)
    for DA3DR in DA3DRs:
        DA3DR.try_recover(cooldown=DRONE_RECOVERY_DELAY)

    # Update and clean up targets
    fixed_targets[:] = [t for t in fixed_targets if t.update_size_and_lifetime()]
    moving_targets[:] = [t for t in moving_targets if t.update_size_and_lifetime()]

    # Draw fixed targets
    for t in fixed_targets:
        glPushMatrix()
        glTranslatef(*t.pos)
        glColor3f(0, 1, 0) if t.active else glColor3f(0, 0, 0)
        draw_sphere(t.size, 20, 20)
        glPopMatrix()

    # Draw moving targets
    for mt in moving_targets:
        glPushMatrix()
        glTranslatef(*mt.pos)
        glColor3f(0, 0.6, 0) if mt.active else glColor3f(0, 0, 0)
        draw_sphere(mt.size, 20, 20)
        glPopMatrix()

    # Update moving targets (movement and status)
    for mt in moving_targets:
        mt.update()
        mt.update_status(DA3DRs)

    # Update fixed targets (status)
    for t in fixed_targets:
        t.update_status(DA3DRs)

    # If all targets are inactive, spawn new moving targets
    if all_targets_inactive(all_targets):
        if not new_targets_spawned:
            n_new_targets = random.randint(1, nbRandomTarget)
            for _ in range(n_new_targets):
                start_z = random.uniform(-3 * SPACE_SIZE, -2 * SPACE_SIZE)
                start_x = random.uniform(-SPACE_SIZE/2, SPACE_SIZE/2)
                start_y = 0
                new_target = MovingTarget([start_x, start_y, start_z])
                moving_targets.append(new_target)
                all_targets.append(new_target)
                stats["active_targets"] += 1
            new_targets_spawned = True
    else:
        new_targets_spawned = False

    # Assign drones to active moving targets
    assign_DA3DRs_to_moving_targets(DA3DRs, moving_targets)

    # Drones select and move toward targets
    for DA3DR in DA3DRs:
        if not DA3DR.active:
            continue  # Skip destroyed drones
        if DA3DR.target is None or (DA3DR.target and not DA3DR.target.active):
            DA3DR.choose_target([t for t in all_targets if t.active and t not in moving_targets])
        DA3DR.move_towards()
        DA3DR.draw()

    # Draw statistics overlay
    draw_stats()

def assign_DA3DRs_to_moving_targets(DA3DRs, moving_targets):
    active_targets = [t for t in moving_targets if t.active]
    n_targets = len(active_targets)
    if n_targets == 0:
        for DA3DR in DA3DRs:
            DA3DR.target = None
        return
    for i, DA3DR in enumerate(DA3DRs):
        if not DA3DR.active:
            continue
        target_index = i % n_targets
        DA3DR.target = active_targets[target_index]

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
