# Author(s): Dr. Patrick Lemoine
# Simulation: Collective Robotics Defense with AI

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation

import numpy as np

# Todo : - Next step Store these parameters in a JSON file so that other scenarios can be loaded.
#        - Perform the visualization using OpenGL or Unity3D – to be decided.
#        - Improve the AI ​​and tactics, add a learning model, and enhance group behavior.
#        - See how to integrate this with real drones.

# ============================================================
# CONFIG SCENARIO / WEATHER / VISU
# ============================================================


CONFIG = {
    # --- Scenario ---
    "NUM_BLUE_DRONES": 10,
    "NUM_RED_DRONES": 10,
    "MAX_TURNS": 5000,
    "MAP_WIDTH": 100,
    "MAP_HEIGHT": 100,

    # --- Drones ---
    "PLAYER_MAX_SPEED": 20.0,
    "PLAYER_MAX_ACCEL": 10.0,
    "ENEMY_MAX_SPEED": 20.0,
    "ENEMY_MAX_ACCEL": 10.0,
    "LOW_HP_THRESHOLD": 30.0,   

    # --- Volume / Altitude ---
    "MIN_ALTITUDE": 0.0,
    "MAX_ALTITUDE_TEAM": {      # 1 = player, 2 = enemy
        1: 150.0,
        2: 200.0,
    },
    "NFZ_LIST": [
        # (center_x, center_y, radius, z_min, z_max)
        (50.0, 50.0, 15.0, 0.0, 300.0),
    ],

    # --- Meteo per label ---
    "WEATHER_PRESETS": {
        "CLEAR": {
            "wind": (0.5, 0.0, 0.0),
            "turb_xy": 0.1,
            "turb_z": 0.05,
            "battery_factor": 1.0,
        },
        "MEDIUM": {
            "wind": (1.5, -0.5, 0.0),
            "turb_xy": 0.3,
            "turb_z": 0.1,
            "battery_factor": 1.2,
        },
        "BAD": {
            "wind": (3.0, -1.0, 0.0),
            "turb_xy": 0.5,
            "turb_z": 0.2,
            "battery_factor": 1.4,
        },
    },

    # --- Visualisation / Heatmaps ---
    "HEATMAP_TURNS": None,   # None => début / milieu / fin
}


# ============================================================
# 0. General utilities
# ============================================================

def angle_wrap(angle: float) -> float:
    """Garde un angle dans [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ============================================================
# 1. Core structures: drones + campaign
# ============================================================

class DroneStatus(Enum):
    ACTIVE = "ACTIVE"
    DESTROYED = "DESTROYED"  # kept for completeness
    OUT_OF_BATTERY = "OUT_OF_BATTERY"
    INTERCEPTED = "INTERCEPTED"  # shot down and crashed to the ground


@dataclass
class DroneUnit:
    """Basic drone representation for the drone battle scenario.
    Teams use "player" / "enemy".
    """
    id: str
    team: str  # "player" or "enemy"
    x: float
    y: float
    z: float = 0.0  # altitude

    hp: float = 100.0
    battery: float = 1000.0  # electric battery charge
    ammo: int = 4

    sensor_range: float = 30.0
    weapon_range: float = 20.0
    
    max_speed: float = 5.0

    status: DroneStatus = DroneStatus.ACTIVE

    # Cinematics
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0

    # Dynamic parameters
    dyn_max_speed: float = 20.0      # maximum horizontal speed (m/s)
    dyn_max_accel: float = 10.0      # accel max (m/s^2)
    dyn_max_yaw_rate: float = math.radians(45.0)  # rad/s
    climb_speed_factor: float = 0.5  # ratio vertical / horizontal

    # Battery / power consumption
    battery_level: float = 1.0
    base_consumption: float = 0.001
    speed_consumption: float = 0.00005
    climb_consumption: float = 0.0001

    def is_alive(self) -> bool:
        return self.status == DroneStatus.ACTIVE and self.hp > 0

    def distance_to(self, other: "DroneUnit") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass
class DroneScenarioState:
    """State holder for the drone scenario."""
    drones: List[DroneUnit] = field(default_factory=list)
    turn_index: int = 0
    max_turns: int = 500

    weather: str = "CLEAR"  # CLEAR / MEDIUM / BAD

    map_width: int = CONFIG["MAP_WIDTH"]
    map_height: int = CONFIG["MAP_HEIGHT"]

    influence_grid_player: List[List[float]] = field(default_factory=list)
    influence_grid_enemy: List[List[float]] = field(default_factory=list)

    # history[turn][drone_id] = (x, y, z, hp, battery, status)
    history: List[Dict[str, Tuple[float, float, float, float, float, DroneStatus]]] = field(
        default_factory=list
    )

    # Shooting/battery/interception statistics
    shot_events: List[Dict[str, Any]] = field(default_factory=list)

    def init_influence_grids(self, grid_size: int = 20) -> None:
        self.influence_grid_player = [
            [0.0 for _ in range(grid_size)] for _ in range(grid_size)
        ]
        self.influence_grid_enemy = [
            [0.0 for _ in range(grid_size)] for _ in range(grid_size)
        ]

    def get_team_drones(self, team: str) -> List[DroneUnit]:
        return [d for d in self.drones if d.team == team and d.is_alive()]

    def is_battle_over(self) -> Optional[str]:
        """Returns:
        "player" : player wins
        "enemy" : enemy wins
        "draw"  : both destroyed
        "timeout" : max turns reached
        None   : battle continues
        """
        player_alive = any(d.is_alive() for d in self.get_team_drones("player"))
        enemy_alive = any(d.is_alive() for d in self.get_team_drones("enemy"))

        if not player_alive and not enemy_alive:
            return "draw"
        if not player_alive:
            return "enemy"
        if not enemy_alive:
            return "player"
        if self.turn_index >= self.max_turns:
            return "timeout"
        return None


@dataclass
class CampaignState:
    """Minimal campaign state compatible with the drone scenario."""
    turn: int = 0
    weather: str = "CLEAR"
    logs: List[str] = field(default_factory=list)

    enemy_ai: "EnhancedEnemyAI" = None
    drone_scenario: Optional[DroneScenarioState] = None
    drone_mode_enabled: bool = False

    player_controls: Dict[str, Any] = field(default_factory=dict)

    def log(self, msg: str) -> None:
        line = f"[Turn {self.turn:03d}] {msg}"
        print(line)
        self.logs.append(line)


# ============================================================
# 2. Volume constraints + Weather
# ============================================================

class VolumeConstraints:
    def __init__(self):
        self.min_altitude = CONFIG["MIN_ALTITUDE"]
        # plafond par équipe: 1=player, 2=enemy
        self.max_altitude_team: Dict[int, float] = CONFIG["MAX_ALTITUDE_TEAM"].copy()
        # Cylindrical NFZ zones
        self.no_fly_zones: List[Dict[str, Any]] = []
       
        for cx, cy, radius, z_min, z_max in CONFIG["NFZ_LIST"]:
            self.add_nfz_cylinder(center=(cx, cy),
                                  radius=radius,
                                  z_min=z_min,
                                  z_max=z_max)

    def add_nfz_cylinder(self, center, radius, z_min=0.0, z_max=1000.0):
        self.no_fly_zones.append(
            {
                "center": np.array(center, dtype=float),
                "radius": float(radius),
                "z_min": float(z_min),
                "z_max": float(z_max),
            }
        )


class Weather:
    def __init__(self):
        # wind in m/s (xy, z)
        self.wind_vector = np.array([0.0, 0.0, 0.0], dtype=float)
        self.turbulence_std_xy = 0.0
        self.turbulence_std_z = 0.0
        self.battery_drain_factor = 1.0

    def sample_turbulence(self) -> np.ndarray:
        d_vx = np.random.normal(0.0, self.turbulence_std_xy)
        d_vy = np.random.normal(0.0, self.turbulence_std_xy)
        d_vz = np.random.normal(0.0, self.turbulence_std_z)
        return np.array([d_vx, d_vy, d_vz], dtype=float)


def apply_volume_constraints(drone: DroneUnit, constraints: VolumeConstraints):
    # map team string -> index (1 ou 2)
    team_id = 1 if drone.team == "player" else 2
    max_alt = constraints.max_altitude_team.get(team_id, None)

    if drone.z < constraints.min_altitude:
        drone.z = constraints.min_altitude
        drone.vz = max(0.0, drone.vz)

    if max_alt is not None and drone.z > max_alt:
        drone.z = max_alt
        drone.vz = min(0.0, drone.vz)

    pos_xy = np.array([drone.x, drone.y], dtype=float)
    for nfz in constraints.no_fly_zones:
        if nfz["z_min"] <= drone.z <= nfz["z_max"]:
            d = np.linalg.norm(pos_xy - nfz["center"])
            if d < nfz["radius"]:
                if d == 0:
                    direction = np.array([1.0, 0.0], dtype=float)
                else:
                    direction = (pos_xy - nfz["center"]) / d
                pos_xy = nfz["center"] + direction * nfz["radius"]
                drone.x, drone.y = float(pos_xy[0]), float(pos_xy[1])

                vxy = np.array([drone.vx, drone.vy], dtype=float)
                radial_speed = np.dot(vxy, -direction)
                if radial_speed > 0:
                    vxy = vxy + radial_speed * direction
                    drone.vx, drone.vy = float(vxy[0]), float(vxy[1])


def apply_weather(drone: DroneUnit, weather: Weather, dt: float):
    # wind = position drift + turbulence on speed
    drone.x += float(weather.wind_vector[0]) * dt
    drone.y += float(weather.wind_vector[1]) * dt
    drone.z += float(weather.wind_vector[2]) * dt

    d_v = weather.sample_turbulence()
    drone.vx += float(d_v[0])
    drone.vy += float(d_v[1])
    drone.vz += float(d_v[2])


def update_battery(drone: DroneUnit, weather: Weather, dt: float):
    speed_xy = math.hypot(drone.vx, drone.vy)
    speed_z = abs(drone.vz)
    cons = (
        drone.base_consumption
        + drone.speed_consumption * speed_xy
        + drone.climb_consumption * speed_z
    )
    cons *= weather.battery_drain_factor
    drone.battery_level -= cons * dt
    drone.battery_level = max(drone.battery_level, 0.0)

    # To maintain compatibility with the "battery" field
    drone.battery -= cons * dt * 1000.0
    drone.battery = max(drone.battery, 0.0)

    if drone.battery <= 0 and drone.status == DroneStatus.ACTIVE:
        drone.status = DroneStatus.OUT_OF_BATTERY

    # Gradual performance reduction if battery is low
    if drone.battery_level < 0.2:
        drone.dyn_max_speed = max(5.0, drone.dyn_max_speed * 0.99)
        drone.dyn_max_accel = max(2.0, drone.dyn_max_accel * 0.99)


# ============================================================
#3. Drone dynamics (inertia, acceleration, yaw)
# ============================================================

def update_drone_dynamics(drone: DroneUnit,
                          desired_vel: Tuple[float, float, float],
                          dt: float):
    """
    desired_vel: Target velocity vector (vx, vy, vz) coming from the AI. 
    A simple dynamic model is applied with limited acceleration
    and maximum angular velocity (yaw_rate).
    """

    # Target orientation given by the horizontal component of desired_vel
    dvx, dvy, dvz = desired_vel
    if abs(dvx) + abs(dvy) > 1e-6:
        desired_yaw = math.atan2(dvy, dvx)
        yaw_error = angle_wrap(desired_yaw - drone.yaw)
        desired_yaw_rate = yaw_error / dt
        # saturation yaw_rate
        desired_yaw_rate = max(
            -drone.dyn_max_yaw_rate,
            min(drone.dyn_max_yaw_rate, desired_yaw_rate),
        )
        drone.yaw_rate = desired_yaw_rate
        drone.yaw = angle_wrap(drone.yaw + drone.yaw_rate * dt)

    # Target velocity in XY aligned with yaw
    desired_speed_xy = math.hypot(dvx, dvy)
    desired_speed_xy = min(desired_speed_xy, drone.dyn_max_speed)
    dir_xy = np.array([math.cos(drone.yaw), math.sin(drone.yaw)], dtype=float)
    vxy_target = dir_xy * desired_speed_xy

    vxy = np.array([drone.vx, drone.vy], dtype=float)
    dvxy = vxy_target - vxy

    norm_dv = np.linalg.norm(dvxy)
    max_dv = drone.dyn_max_accel * dt
    if norm_dv > max_dv > 0:
        dvxy = dvxy * (max_dv / norm_dv)

    vxy = vxy + dvxy

    speed_xy = np.linalg.norm(vxy)
    if speed_xy > drone.dyn_max_speed:
        vxy = vxy * (drone.dyn_max_speed / speed_xy)

    drone.vx, drone.vy = float(vxy[0]), float(vxy[1])

    # Vertical component (slower)
    desired_vz = max(-drone.dyn_max_speed * drone.climb_speed_factor,
                     min(drone.dyn_max_speed * drone.climb_speed_factor, dvz))
    dvz_cmd = desired_vz - drone.vz
    max_dvz = drone.dyn_max_accel * dt * drone.climb_speed_factor
    dvz_cmd = max(-max_dvz, min(max_dvz, dvz_cmd))
    drone.vz += dvz_cmd

    # Position integration
    drone.x += drone.vx * dt
    drone.y += drone.vy * dt
    drone.z += drone.vz * dt

    # Ground clamp
    if drone.is_alive() and drone.z < 0.0:
        drone.z = 0.0
        if drone.vz < 0.0:
            drone.vz = 0.0


# ============================================================
# 4. Drone interception helper
# ============================================================

def drone_intercepted(drone: DroneUnit, campaign_state: Optional[CampaignState] = None) -> None:
    """Handle interception: drone is shot down, falls to the ground and becomes 'INTERCEPTED'."""
    drone.hp = 0.0
    drone.status = DroneStatus.INTERCEPTED
    drone.z = 0.0  # on the ground
    drone.vx = drone.vy = drone.vz = 0.0
    if campaign_state is not None:
        campaign_state.log(f"[Drone] {drone.id} intercepted and crashed to the ground")


# ============================================================
#5. EnhancedEnemyAI (simplified core + drone extension)
# ============================================================

class EnhancedEnemyAI:
    def __init__(self):
        self.personality: str = "deceptive"  # "aggressive" / "defensive" / "deceptive"
        self.memory: List[Dict[str, Any]] = []
        self.memory_size: int = 5

    def observe_outcome(self, outcome: Dict[str, Any]) -> None:
        self.memory.append(outcome)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def decide_personality(self) -> None:
        if not self.memory:
            return
        wins = sum(1 for o in self.memory if o["winner_team"] == "enemy")
        win_rate = wins / len(self.memory)
        if win_rate > 0.7:
            self.personality = "aggressive"
        elif win_rate < 0.3:
            self.personality = "defensive"
        else:
            self.personality = "deceptive"

    def recommend_drone_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        if not drone.is_alive():
            return (0.0, 0.0, 0.0), None

        enemies = state.get_team_drones("player" if drone.team == "enemy" else "enemy")
        if not enemies:
            return (0.0, 0.0, 0.0), None

        target = min(enemies, key=lambda e: drone.distance_to(e))
        dist = drone.distance_to(target)

        if self.personality == "aggressive":
            vx, vy, vz = self._vector_towards(drone, target, factor=1.0)
            shoot = target if (dist <= drone.weapon_range and drone.ammo > 0) else None
        elif self.personality == "defensive":
            desired = drone.weapon_range * 0.7
            if dist < desired:
                vx, vy, vz = self._vector_away(drone, target, factor=1.0)
            else:
                vx, vy, vz = self._vector_towards(drone, target, factor=0.5)
            shoot = target if (dist <= drone.weapon_range * 0.9 and drone.ammo > 0) else None
        else:  # deceptive
            if drone.hp < 40:
                vx, vy, vz = self._vector_away(drone, target, factor=1.0)
            else:
                vx, vy, vz = self._flank_vector(drone, target)
            shoot = target if (dist <= drone.weapon_range * 0.8 and drone.ammo > 0) else None

        return (vx, vy, vz), shoot

    def _vector_towards(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = target.x - drone.x, target.y - drone.y, target.z - drone.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _vector_away(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = drone.x - target.x, drone.y - target.y, drone.z - target.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _flank_vector(self, drone: DroneUnit, target: DroneUnit):
        dx, dy = target.x - drone.x, target.y - drone.y
        norm = math.hypot(dx, dy) + 1e-6
        fx, fy = -dy / norm, dx / norm
        return (
            fx * drone.dyn_max_speed,
            fy * drone.dyn_max_speed,
            0.0,
        )


# ============================================================
# 6. ChessSunTzuAI + GoSunTzuAI
# ============================================================

class ChessSunTzuAI:
    def __init__(self, map_width: int = 100, map_height: int = 100):
        self.map_width = map_width
        self.map_height = map_height

    def positional_score_for_team(self, state: DroneScenarioState, team: str) -> float:
        drones = state.get_team_drones(team)
        if not drones:
            return -999.0
        score = 0.0
        cx, cy = self.map_width / 2.0, self.map_height / 2.0
        for d in drones:
            dist_center = math.hypot(d.x - cx, d.y - cy)
            score += max(0.0, 50.0 - dist_center)
            score += d.hp * 0.5
            margin = min(d.x, d.y, self.map_width - d.x, self.map_height - d.y)
            score += max(-30.0, margin - 30.0)
        return score


class GoSunTzuAI:
    def __init__(self, grid_size: int = 20, map_width: int = 100, map_height: int = 100):
        self.grid_size = grid_size
        self.map_width = map_width
        self.map_height = map_height

    def _cell_center(self, i: int, j: int) -> Tuple[float, float]:
        sx = self.map_width / self.grid_size
        sy = self.map_height / self.grid_size
        return (i + 0.5) * sx, (j + 0.5) * sy

    def compute_influence(self, state: DroneScenarioState) -> None:
        state.init_influence_grids(grid_size=self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cx, cy = self._cell_center(i, j)
                player_infl = 0.0
                enemy_infl = 0.0
                for d in state.drones:
                    if not d.is_alive():
                        continue
                    dist = math.hypot(d.x - cx, d.y - cy)
                    sigma = d.sensor_range
                    if sigma > 0:
                        infl = math.exp(-(dist ** 2) / (2 * sigma ** 2)) * (1.0 + d.hp / 100.0)
                        if d.team == "player":
                            player_infl += infl
                        else:
                            enemy_infl += infl
                state.influence_grid_player[i][j] = player_infl
                state.influence_grid_enemy[i][j] = enemy_infl

    def detect_encirclement_zones(self, state: DroneScenarioState, target_team: str = "enemy") -> List[DroneUnit]:
        if not state.influence_grid_player:
            self.compute_influence(state)
        sx = self.map_width / self.grid_size
        sy = self.map_height / self.grid_size

        encircled: List[DroneUnit] = []
        for d in state.drones:
            if not d.is_alive() or d.team != target_team:
                continue
            i = int(d.x / sx)
            j = int(d.y / sy)
            if i < 0 or j < 0 or i >= self.grid_size or j >= self.grid_size:
                continue
            player_sum = 0.0
            enemy_sum = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.grid_size and 0 <= jj < self.grid_size:
                        player_sum += state.influence_grid_player[ii][jj]
                        enemy_sum += state.influence_grid_enemy[ii][jj]
            if target_team == "enemy":
                if player_sum > 2.0 * max(1e-6, enemy_sum):
                    encircled.append(d)
            else:
                if enemy_sum > 2.0 * max(1e-6, player_sum):
                    encircled.append(d)
        return encircled


# ============================================================
# 7. PlayerDroneController
# ============================================================

class PlayerDroneController:
    def __init__(self, controls: Dict[str, Any]):
        self.controls = controls

    def _get_personality(self) -> str:
        return self.controls.get("player_personality", "defensive")

    def _get_aggressiveness(self) -> float:
        return float(self.controls.get("player_aggressiveness", 0.5))

    def _get_pref_distance(self, default: float) -> float:
        return float(self.controls.get("player_pref_distance", default))

    def recommend_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        enemies = state.get_team_drones("enemy")
        if not drone.is_alive() or not enemies:
            return (0.0, 0.0, 0.0), None

        target = min(enemies, key=lambda e: drone.distance_to(e))
        dist = drone.distance_to(target)

        personality = self._get_personality()
        aggr = self._get_aggressiveness()
        pref_dist = self._get_pref_distance(drone.weapon_range * 0.8)

        if personality == "aggressive":
            factor = 0.5 + aggr * 0.5
            vx, vy, vz = self._vector_towards(drone, target, factor=factor)
            shoot = target if (dist <= drone.weapon_range and drone.ammo > 0) else None
        elif personality == "defensive":
            if dist < pref_dist:
                vx, vy, vz = self._vector_away(drone, target, factor=0.5 + (1.0 - aggr) * 0.5)
            else:
                vx, vy, vz = self._vector_towards(drone, target, factor=0.3 + aggr * 0.2)
            shoot = target if (dist <= drone.weapon_range * 0.9 and drone.ammo > 0) else None
        else:  # deceptive
            if drone.hp < 40:
                vx, vy, vz = self._vector_away(drone, target, factor=0.7 + (1.0 - aggr) * 0.3)
            else:
                vx, vy, vz = self._flank_vector(drone, target)
            shoot = target if (dist <= drone.weapon_range * 0.8 and drone.ammo > 0) else None

        return (vx, vy, vz), shoot

    def _vector_towards(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = target.x - drone.x, target.y - drone.y, target.z - drone.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _vector_away(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = drone.x - target.x, drone.y - target.y, drone.z - target.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _flank_vector(self, drone: DroneUnit, target: DroneUnit):
        dx, dy = target.x - drone.x, target.y - drone.y
        norm = math.hypot(dx, dy) + 1e-6
        fx, fy = -dy / norm, dx / norm
        return (
            fx * drone.dyn_max_speed,
            fy * drone.dyn_max_speed,
            0.0,
        )


# ============================================================
# 8. Drone initialization helper (variable counts)
# ============================================================

def create_drones(num_blue: int, num_red: int) -> List[DroneUnit]:
    """Creates num_blue player drones (left side) and num_red enemy drones (right side),
    spaced in Y and slightly in Z.
    """
    drones: List[DroneUnit] = []

    # Blue drones (player) on the left
    start_x_blue = 10.0
    base_y_blue = 40.0

    for i in range(num_blue):
        y = base_y_blue + i * 8.0
        z = 5.0 + i * 1.5
        drones.append(
            DroneUnit(
                id=f"B{i+1}",
                team="player",
                x=start_x_blue,
                y=y,
                z=z,
                hp=120.0,
                battery=1200.0,
                sensor_range=30.0,
                weapon_range=20.0,
                max_speed=5.0,
                dyn_max_speed=CONFIG["PLAYER_MAX_SPEED"],
                dyn_max_accel=CONFIG["PLAYER_MAX_ACCEL"],
                battery_level=1.0,
            )
        )

    # Red drones (enemy) on the right
    start_x_red = 90.0
    base_y_red = 40.0

    for j in range(num_red):
        y = base_y_red + j * 8.0
        z = 8.0 + j * 1.5
        drones.append(
            DroneUnit(
                id=f"R{j+1}",
                team="enemy",
                x=start_x_red,
                y=y,
                z=z,
                hp=100.0,
                battery=1200.0,
                sensor_range=35.0 if j == 0 else 30.0,
                weapon_range=22.0 if j == 0 else 20.0,
                max_speed=5.0,
                dyn_max_speed=CONFIG["ENEMY_MAX_SPEED"],
                dyn_max_accel=CONFIG["ENEMY_MAX_ACCEL"],
                battery_level=1.0,
            )
        )

    return drones


# ============================================================
# 9. Collision avoidance helper
# ============================================================

def apply_collision_avoidance(
    ds: DroneScenarioState, min_dist: float = 3.0, strength: float = 1.5
) -> Dict[str, Tuple[float, float, float]]:
    """Simple collision avoidance:
    - For each pair of alive drones closer than min_dist, apply a repulsive vector.
    - Returns a dict: extra_vel[drone_id] = (dvx, dvy, dvz)
    """
    extra_vel: Dict[str, List[float]] = {
        d.id: [0.0, 0.0, 0.0] for d in ds.drones if d.is_alive()
    }

    drones_alive = [d for d in ds.drones if d.is_alive()]
    for i in range(len(drones_alive)):
        for j in range(i + 1, len(drones_alive)):
            a = drones_alive[i]
            b = drones_alive[j]
            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
            if dist < min_dist:
                factor = strength * (min_dist - dist) / dist
                rx, ry, rz = dx * factor, dy * factor, dz * factor
                extra_vel[a.id][0] += rx
                extra_vel[a.id][1] += ry
                extra_vel[a.id][2] += rz
                extra_vel[b.id][0] -= rx
                extra_vel[b.id][1] -= ry
                extra_vel[b.id][2] -= rz

    return {k: (v[0], v[1], v[2]) for k, v in extra_vel.items()}


# ============================================================
# 10. Export shooting/battery/interception stats
# ============================================================

def export_battle_stats(ds: DroneScenarioState, campaign_state: CampaignState) -> None:
    distances_player = [e["distance"] for e in ds.shot_events if e["shooter_team"] == "player"]
    distances_enemy = [e["distance"] for e in ds.shot_events if e["shooter_team"] == "enemy"]

    hits_player = [e for e in ds.shot_events if e["shooter_team"] == "player" and e["hit"]]
    hits_enemy = [e for e in ds.shot_events if e["shooter_team"] == "enemy" and e["hit"]]

    total_player = len([e for e in ds.shot_events if e["shooter_team"] == "player"])
    total_enemy = len([e for e in ds.shot_events if e["shooter_team"] == "enemy"])

    hit_rate_player = len(hits_player) / total_player if total_player > 0 else 0.0
    hit_rate_enemy = len(hits_enemy) / total_enemy if total_enemy > 0 else 0.0

    # Average battery life (we're using the last snapshot)
    if ds.history:
        last_snap = ds.history[-1]
        final_batteries_player = [
            last_snap[did][4] for did in last_snap.keys() if did.startswith("B")
        ]
        final_batteries_enemy = [
            last_snap[did][4] for did in last_snap.keys() if did.startswith("R")
        ]
    else:
        final_batteries_player = []
        final_batteries_enemy = []

    avg_batt_player = (
        sum(final_batteries_player) / len(final_batteries_player)
        if final_batteries_player
        else 0.0
    )
    avg_batt_enemy = (
        sum(final_batteries_enemy) / len(final_batteries_enemy)
        if final_batteries_enemy
        else 0.0
    )

    interception_turns_player = []
    interception_turns_enemy = []

    for d in ds.drones:
        if d.status == DroneStatus.INTERCEPTED:
            first_turn = None
            for t, snap in enumerate(ds.history):
                if snap[d.id][5] == DroneStatus.INTERCEPTED:
                    first_turn = t
                    break
            if first_turn is not None:
                if d.team == "player":
                    interception_turns_player.append(first_turn)
                else:
                    interception_turns_enemy.append(first_turn)

    avg_turn_intercept_player = (
        sum(interception_turns_player) / len(interception_turns_player)
        if interception_turns_player
        else 0.0
    )
    avg_turn_intercept_enemy = (
        sum(interception_turns_enemy) / len(interception_turns_enemy)
        if interception_turns_enemy
        else 0.0
    )

    campaign_state.log(
        f"[Stats] Player hit rate={hit_rate_player:.2f}, Enemy hit rate={hit_rate_enemy:.2f}"
    )
    campaign_state.log(
        f"[Stats] Avg battery P={avg_batt_player:.1f}, E={avg_batt_enemy:.1f}"
    )
    campaign_state.log(
        f"[Stats] Avg turns before intercept P={avg_turn_intercept_player:.1f}, "
        f"E={avg_turn_intercept_enemy:.1f}"
    )

    # Export CSV basique
    try:
        with open("shot_events.csv", "w", encoding="utf-8") as f:
            f.write("turn,shooter_id,team,target_id,hit,distance,damage\n")
            for e in ds.shot_events:
                f.write(
                    f"{e['turn']},{e['shooter_id']},{e['shooter_team']},"
                    f"{e['target_id']},{int(e['hit'])},{e['distance']:.3f},"
                    f"{e['damage']:.1f}\n"
                )
    except Exception as ex:
        campaign_state.log(f"[Stats] Could not write shot_events.csv: {ex}")


# ============================================================
# 11. Main loop: run_drone_campaign
# ============================================================

def run_drone_campaign(
    campaign_state: CampaignState,
    num_blue: int = 2,
    num_red: int = 3,
    max_turns: int = 100,
) -> None:
    ds = DroneScenarioState(
        max_turns=max_turns,
        weather=campaign_state.weather,
        map_width=CONFIG["MAP_WIDTH"],
        map_height=CONFIG["MAP_HEIGHT"],
    )
    ds.drones = create_drones(num_blue=num_blue, num_red=num_red)
    ds.init_influence_grids()

    chess_ai = ChessSunTzuAI(map_width=ds.map_width, map_height=ds.map_height)
    go_ai = GoSunTzuAI(grid_size=20, map_width=ds.map_width, map_height=ds.map_height)

    enemy_ai: EnhancedEnemyAI = campaign_state.enemy_ai
    player_ai = PlayerDroneController(campaign_state.player_controls)

    # Volume & Météo
    constraints = VolumeConstraints()

    weather = Weather()
    preset = CONFIG["WEATHER_PRESETS"].get(
        campaign_state.weather, CONFIG["WEATHER_PRESETS"]["CLEAR"]
    )
    weather.wind_vector = np.array(preset["wind"], dtype=float)
    weather.turbulence_std_xy = preset["turb_xy"]
    weather.turbulence_std_z = preset["turb_z"]
    weather.battery_drain_factor = preset["battery_factor"]

    campaign_state.log(
        f"=== Drone Campaign: {num_red} enemy drones vs {num_blue} player drones ==="
    )

    dt = 1.0  # pas de temps

    while ds.turn_index < ds.max_turns:
        ds.turn_index += 1
        campaign_state.turn += 1

        result = ds.is_battle_over()
        if result is not None:
            campaign_state.log(f"[Drone] Battle result: {result}")
            if result == "enemy":
                winner_team = "enemy"
            elif result == "player":
                winner_team = "player"
            else:
                winner_team = "draw"
            outcome = {
                "winner_team": winner_team,
                "player_alive": len(ds.get_team_drones("player")),
                "enemy_alive": len(ds.get_team_drones("enemy")),
                "turns": ds.turn_index,
            }
            enemy_ai.observe_outcome(outcome)
            enemy_ai.decide_personality()
            break

        chess_enemy = chess_ai.positional_score_for_team(ds, "enemy")
        chess_player = chess_ai.positional_score_for_team(ds, "player")
        go_ai.compute_influence(ds)
        encircled_player = go_ai.detect_encirclement_zones(ds, target_team="player")

        campaign_state.log(
            f"[DroneChess] enemy_pos={chess_enemy:.1f}, player_pos={chess_player:.1f}"
        )
        if encircled_player:
            names = ", ".join(d.id for d in encircled_player)
            campaign_state.log(f"[DroneGo] Player drones encircled: {names}")

        planned_moves: Dict[str, Tuple[float, float, float]] = {}
        planned_shots: List[Tuple[DroneUnit, DroneUnit]] = []

        for d in ds.drones:
            if not d.is_alive():
                continue

            if d.team == "enemy":
                desired_vel, target = enemy_ai.recommend_drone_action(d, ds)
            else:
                desired_vel, target = player_ai.recommend_action(d, ds)

            planned_moves[d.id] = desired_vel
            if target is not None:
                planned_shots.append((d, target))

        extra_vel = apply_collision_avoidance(ds, min_dist=3.0, strength=1.5)

        # Movement + dynamics + weather + volume + battery
        for d in ds.drones:
            if not d.is_alive():
                continue

            base_vel = planned_moves.get(d.id, (0.0, 0.0, 0.0))
            if d.id in extra_vel:
                evx, evy, evz = extra_vel[d.id]
                base_vel = (
                    base_vel[0] + evx,
                    base_vel[1] + evy,
                    base_vel[2] + evz,
                )

            update_drone_dynamics(d, base_vel, dt)
            apply_weather(d, weather, dt)
            apply_volume_constraints(d, constraints)
            update_battery(d, weather, dt)

         # Firing phase + enregistrement stats
        for shooter, target in planned_shots:
            if not shooter.is_alive() or not target.is_alive():
                continue
            if shooter.ammo <= 0:
                continue

            dist = shooter.distance_to(target)
            if dist > shooter.weapon_range:
                continue

            hit_prob = max(0.5, 1.5 - dist / (shooter.weapon_range + 1e-6))
            if ds.weather == "MEDIUM":
                hit_prob *= 0.8
            elif ds.weather == "BAD":
                hit_prob *= 0.6

            hit = random.random() < hit_prob
            damage = 0.0

            if hit:
                dmg = random.uniform(60, 120)
                damage = dmg
                target.hp -= dmg
                campaign_state.log(
                    f"[Drone] {shooter.id} hits {target.id} for {dmg:.1f} HP (dist={dist:.1f})"
                )
                if target.hp <= 0 and target.status == DroneStatus.ACTIVE:
                    drone_intercepted(target, campaign_state)
            else:
                campaign_state.log(
                    f"[Drone] {shooter.id} misses {target.id} (dist={dist:.1f})"
                )

            shooter.ammo -= 1

            ds.shot_events.append(
                {
                    "turn": ds.turn_index,
                    "shooter_id": shooter.id,
                    "shooter_team": shooter.team,
                    "target_id": target.id,
                    "hit": hit,
                    "distance": dist,
                    "damage": damage,
                }
            )

        snapshot: Dict[str, Tuple[float, float, float, float, float, DroneStatus]] = {}
        for d in ds.drones:
            snapshot[d.id] = (d.x, d.y, d.z, d.hp, d.battery, d.status)
        ds.history.append(snapshot)

    export_battle_stats(ds, campaign_state)
    campaign_state.drone_scenario = ds


# ============================================================
# 12. Visualisations
# ============================================================

def visualize_drone_battle_3d(campaign_state: CampaignState) -> None:
    ds = campaign_state.drone_scenario
    if ds is None or not ds.history:
        print("No drone scenario history to visualize.")
        return

    turns = len(ds.history)
    drone_ids = list(ds.history[0].keys())

    xs: Dict[str, List[float]] = {d: [] for d in drone_ids}
    ys: Dict[str, List[float]] = {d: [] for d in drone_ids}
    zs: Dict[str, List[float]] = {d: [] for d in drone_ids}
    hps: Dict[str, List[float]] = {d: [] for d in drone_ids}
    statuses: Dict[str, List[DroneStatus]] = {d: [] for d in drone_ids}
    teams: Dict[str, str] = {}

    for snap in ds.history:
        for d_id, (x, y, z, hp, battery, status) in snap.items():
            xs[d_id].append(x)
            ys[d_id].append(y)
            zs[d_id].append(z)
            hps[d_id].append(hp)
            statuses[d_id].append(status)

    for d in ds.drones:
        teams[d.id] = d.team

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    player_high = ax.scatter([], [], [], c="blue", label="Player (>LOW_HP)")
    player_low = ax.scatter([], [], [], c="cyan", label="Player (<=LOW_HP)")
    enemy_high = ax.scatter([], [], [], c="red", label="Enemy (>LOW_HP)")
    enemy_low = ax.scatter([], [], [], c="orange", label="Enemy (<=LOW_HP)")
    shot_hits = ax.scatter([], [], [], c="green", marker="*", s=50, label="Hit")
    shot_misses = ax.scatter([], [], [], c="yellow", marker="x", s=40, label="Miss")
    intercepted_scatter = ax.scatter([], [], [], c="black", label="Intercepted")

    ax.set_xlim(0, ds.map_width)
    ax.set_ylim(0, ds.map_height)
    ax.set_zlim(0, 25)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (altitude)")
    ax.set_title("3D Drone Battle (Red vs Blue)")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=45)

    info_text = ax.text(
        0,
        0,
        30,
        "",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    def update(frame: int):
        phx, phy, phz = [], [], []
        plx, ply, plz = [], [], []
        ehx, ehy, ehz = [], [], []
        elx, ely, elz = [], [], []
        ix, iy, iz = [], [], []
        hit_x, hit_y, hit_z = [], [], []
        miss_x, miss_y, miss_z = [], [], []

        alive_count = 0
        red_intercepted = 0
        blue_intercepted = 0

        for d_id in drone_ids:
            if frame >= len(xs[d_id]):
                continue
            x = xs[d_id][frame]
            y = ys[d_id][frame]
            z = zs[d_id][frame]
            hp = hps[d_id][frame]
            st = statuses[d_id][frame]
            team = teams[d_id]

            if st == DroneStatus.INTERCEPTED:
                ix.append(x)
                iy.append(y)
                iz.append(z)
                if team == "enemy":
                    red_intercepted += 1
                else:
                    blue_intercepted += 1
            else:
                if st == DroneStatus.ACTIVE:
                    alive_count += 1
                if team == "player":
                    if hp <= CONFIG["LOW_HP_THRESHOLD"]:
                        plx.append(x)
                        ply.append(y)
                        plz.append(z)
                    else:
                        phx.append(x)
                        phy.append(y)
                        phz.append(z)
                else:
                    if hp <= CONFIG["LOW_HP_THRESHOLD"]:
                        elx.append(x)
                        ely.append(y)
                        elz.append(z)
                    else:
                        ehx.append(x)
                        ehy.append(y)
                        ehz.append(z)

        # shots in this round (turn_index starts at 1)
        turn_shots = [e for e in ds.shot_events if e["turn"] == frame + 1]
        for e in turn_shots:
            tid = e["target_id"]
            if frame < len(xs[tid]):
                tx = xs[tid][frame]
                ty = ys[tid][frame]
                tz = zs[tid][frame]
                if e["hit"]:
                    hit_x.append(tx)
                    hit_y.append(ty)
                    hit_z.append(tz)
                else:
                    miss_x.append(tx)
                    miss_y.append(ty)
                    miss_z.append(tz)

        player_high._offsets3d = (phx, phy, phz)
        player_low._offsets3d = (plx, ply, plz)
        enemy_high._offsets3d = (ehx, ehy, ehz)
        enemy_low._offsets3d = (elx, ely, elz)
        intercepted_scatter._offsets3d = (ix, iy, iz)
        shot_hits._offsets3d = (hit_x, hit_y, hit_z)
        shot_misses._offsets3d = (miss_x, miss_y, miss_z)

        info_text.set_text(
            f"Alive: {alive_count}\n"
            f"Red intercepted: {red_intercepted}\n"
            f"Blue intercepted: {blue_intercepted}"
        )
        return (
            player_high,
            player_low,
            enemy_high,
            enemy_low,
            intercepted_scatter,
            shot_hits,
            shot_misses,
            info_text,
        )

    anim = FuncAnimation(fig, update, frames=turns, interval=200, blit=False)
    plt.show()


def visualize_drone_2d_tracks(campaign_state: CampaignState) -> None:
    ds = campaign_state.drone_scenario
    if ds is None or not ds.history:
        print("No drone scenario history to visualize.")
        return

    turns = len(ds.history)
    drone_ids = list(ds.history[0].keys())

    xs = {d: [] for d in drone_ids}
    ys = {d: [] for d in drone_ids}
    zs = {d: [] for d in drone_ids}
    teams = {d.id: d.team for d in ds.drones}

    for snap in ds.history:
        for d_id, (x, y, z, hp, battery, status) in snap.items():
            xs[d_id].append(x)
            ys[d_id].append(y)
            zs[d_id].append(z)

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(12, 5))

    # XY
    for d_id in drone_ids:
        if teams[d_id] == "player":
            ax_xy.plot(xs[d_id], ys[d_id], color="blue", alpha=0.7)
        else:
            ax_xy.plot(xs[d_id], ys[d_id], color="red", alpha=0.7)
    ax_xy.set_title("Trajectoires au sol (X–Y)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_xlim(0, ds.map_width)
    ax_xy.set_ylim(0, ds.map_height)

    # altitude vs temps (on trace quelques drones)
    max_to_plot = min(6, len(drone_ids))
    for d_id in drone_ids[:max_to_plot]:
        t_axis = list(range(turns))
        if teams[d_id] == "player":
            ax_z.plot(t_axis, zs[d_id], label=d_id, color="blue", alpha=0.7)
        else:
            ax_z.plot(t_axis, zs[d_id], label=d_id, color="red", alpha=0.7)
    ax_z.set_title("Altitude vs temps")
    ax_z.set_xlabel("Turn")
    ax_z.set_ylabel("Altitude z")
    ax_z.legend()

    plt.tight_layout()
    plt.show()


def visualize_influence_heatmaps(campaign_state: CampaignState,
                                 turns_to_show: Optional[List[int]] = None) -> None:
    ds = campaign_state.drone_scenario
    if ds is None or not ds.history:
        print("No drone scenario history to visualize.")
        return

    if turns_to_show is None:
        cfg_turns = CONFIG["HEATMAP_TURNS"]
        if cfg_turns is not None:
            turns_to_show = cfg_turns
        else:
            T = len(ds.history)
            if T < 3:
                turns_to_show = [0, T // 2, T - 1]
            else:
                turns_to_show = [0, T // 2, T - 1]

    grid_size = len(ds.influence_grid_player) if ds.influence_grid_player else 20
    go_ai = GoSunTzuAI(grid_size=grid_size,
                       map_width=ds.map_width,
                       map_height=ds.map_height)

    fig, axes = plt.subplots(len(turns_to_show), 2,
                             figsize=(8, 3 * len(turns_to_show)))
    if len(turns_to_show) == 1:
        axes = np.array([axes])

    for idx, turn in enumerate(turns_to_show):
        snap_idx = min(turn, len(ds.history) - 1)
        snap = ds.history[snap_idx]

        orig_positions = {d.id: (d.x, d.y, d.z) for d in ds.drones}
        for d in ds.drones:
            x, y, z, hp, battery, status = snap[d.id]
            d.x, d.y, d.z = x, y, z

        go_ai.compute_influence(ds)

        ax_p = axes[idx, 0]
        ax_e = axes[idx, 1]
        im0 = ax_p.imshow(np.array(ds.influence_grid_player).T,
                          origin="lower", cmap="Blues")
        im1 = ax_e.imshow(np.array(ds.influence_grid_enemy).T,
                          origin="lower", cmap="Reds")
        ax_p.set_title(f"Player influence (turn {snap_idx})")
        ax_e.set_title(f"Enemy influence (turn {snap_idx})")
        fig.colorbar(im0, ax=ax_p, fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax_e, fraction=0.046, pad=0.04)

        # restore positions
        for d in ds.drones:
            d.x, d.y, d.z = orig_positions[d.id]

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. Entry point
# ============================================================

if __name__ == "__main__":
    cs = CampaignState()
    cs.enemy_ai = EnhancedEnemyAI()
    # cs.enemy_ai.personality = "aggressive"
    cs.enemy_ai.personality = "defensive"

    cs.player_controls = {
        "player_personality": "defensive",
        "player_aggressiveness": 0.95,
        "player_pref_distance": 10.0,
    }

    cs.drone_mode_enabled = True
    cs.weather = "MEDIUM"  # CLEAR / MEDIUM / BAD

    num_blue_drones = CONFIG["NUM_BLUE_DRONES"]
    num_red_drones = CONFIG["NUM_RED_DRONES"]
    max_turns = CONFIG["MAX_TURNS"]

    if cs.drone_mode_enabled:
        run_drone_campaign(
            cs,
            num_blue=num_blue_drones,
            num_red=num_red_drones,
            max_turns=max_turns,
        )

        visualize_drone_battle_3d(cs)
        visualize_drone_2d_tracks(cs)
        visualize_influence_heatmaps(cs)
