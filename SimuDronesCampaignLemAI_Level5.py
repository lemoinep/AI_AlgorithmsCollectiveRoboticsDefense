# Author(s): Dr. Patrick Lemoine
# Simulation: Collective Robotics Defense with AI and EMP weapons arranged in a circular arc
# + LN2 (liquid nitrogen) Weapon + Kamikaze drones
# For now we will use a swarm of small drones and later we will use AI robot agents
# which will have different characteristics and will be deployed by a strategic AI.

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
import numpy as np


# Todo :
# - Next step Store these parameters in a JSON file so that other scenarios can be loaded.
# - Perform the visualization using OpenGL or Unity3D – to be decided.
# - Improve the AI ​​and tactics, add a learning model, and enhance group behavior.
# - See how to integrate this with real drones.


# ============================================================
# CONFIG SCENARIO / WEATHER / VISU
# ============================================================
# ------------------------------------------------------------

CONFIG = {
    # --- Scenario ---
    "NUM_BLUE_DRONES": 100,
    "NUM_RED_DRONES": 100,
    "MAX_TURNS": 50000,
    "MAP_WIDTH": 500,
    "MAP_HEIGHT": 500,

    # --- EMP / IEM weapons (SPEAR / THOR) ---
    "ENABLE_EMP_WEAPONS": True,
    "PLAYER_EMP_PROB": 0.3,
    "ENEMY_EMP_PROB": 0.3,
    "EMP_RANGE": 35.0,
    "EMP_COOLDOWN_TURNS": 10,
    "EMP_DISABLE_DURATION": 8,
    "EMP_DAMAGE_HP": 15.0,
    "EMP_DAMAGE_BATTERY": 300.0,
    
    # --- Liquid nitrogen spray weapon (short-range, conical) ---
    "ENABLE_LN2_WEAPON": True,
    "LN2_RANGE": 8.0,                  # effective range (meters)
    "LN2_CONE_ANGLE_DEG": 35.0,        # half-angle of cone from drone heading
    "LN2_COOLDOWN_TURNS": 5,
    "LN2_DAMAGE_HP": 10.0,             # structural / component damage
    "LN2_DISABLE_TURNS": 4,            # temporary technical failure
    "LN2_BATTERY_PENALTY": 150.0,      # battery impact / stress
    "LN2_PROB_PLAYER": 0.3,            # probability a blue drone has LN2 module
    "LN2_PROB_ENEMY": 0.2,             # probability a red drone has LN2 module
    
    # --- Kamikaze drones ---
    "ENABLE_KAMIKAZE_WEAPON": True,
    "KAMIKAZE_PROB_PLAYER": 0.2,
    "KAMIKAZE_PROB_ENEMY": 0.2,
    "KAMIKAZE_DAMAGE_HP": 100.0,
    "KAMIKAZE_TRIGGER_RADIUS": 3.0,

    # --- Drones ---
    "PLAYER_MAX_SPEED": 20.0,
    "PLAYER_MAX_ACCEL": 10.0,
    "ENEMY_MAX_SPEED": 20.0,
    "ENEMY_MAX_ACCEL": 10.0,
    "LOW_HP_THRESHOLD": 30.0,
    
    # --- Arc-shaped tactical maneuver around the cluster ---
    "EMP_SAFE_FACTOR": 1.3,        # safety radius = SAFE_FACTOR * EMP_RANGE
    "EMP_MIN_ENEMIES": 3,          # Minimum number of enemies in EMP_RANGE to fire
    "EMP_MAX_ARC_DRONES": 6,       # Maximum number of allied drones on the arc

    # --- Volume / Altitude ---
    "MIN_ALTITUDE": 0.0,
    "MAX_ALTITUDE_TEAM": {  # 1 = player, 2 = enemy
        1: 150.0,
        2: 200.0,
    },

    "NFZ_LIST": [  # (center_x, center_y, radius, z_min, z_max)
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
    "HEATMAP_TURNS": None,  # None => start / mid / end
}

# ============================================================
# 0. General utilities
# ============================================================
# ------------------------------------------------------------

def angle_wrap(angle: float) -> float:
    """Clamp angle in [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ============================================================
# 1. Core structures: drones + campaign
# ============================================================
# ------------------------------------------------------------

class DroneStatus(Enum):
    ACTIVE = "ACTIVE"
    DESTROYED = "DESTROYED"  # kept for completeness
    OUT_OF_BATTERY = "OUT_OF_BATTERY"
    INTERCEPTED = "INTERCEPTED"  # shot down and crashed to the ground


class DroneRole(Enum):
    SCOUT = "SCOUT"   # high speed, information / flanking
    SNIPER = "SNIPER"  # long-range fire, keeps distance
    TANK = "TANK"     # front line, closes distance, soaks damage
    DECOY = "DECOY"   # draws fire, baits encirclement
    KAMIKAZE = "KAMIKAZE"  # sacrificial attack on high-value targets


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

    # Legacy scalar speed; we keep it for compatibility
    max_speed: float = 5.0

    status: DroneStatus = DroneStatus.ACTIVE

    # Cinematics
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0

    # Dynamic parameters
    dyn_max_speed: float = 20.0  # maximum horizontal speed (m/s)
    dyn_max_accel: float = 10.0  # accel max (m/s^2)
    dyn_max_yaw_rate: float = math.radians(45.0)  # rad/s
    climb_speed_factor: float = 0.5  # ratio vertical / horizontal

    # Battery / power consumption
    battery_level: float = 1.0
    base_consumption: float = 0.001
    speed_consumption: float = 0.00005
    climb_consumption: float = 0.0001

    # Strategic role
    role: DroneRole = DroneRole.SCOUT

    # --- EMP / IEM weapon capabilities ---
    has_emp: bool = False
    emp_cooldown: int = 0
    emp_disabled_turns: int = 0
    
    # --- Liquid nitrogen spray module ---
    has_ln2: bool = False
    ln2_cooldown: int = 0
    ln2_disabled_turns: int = 0
    
    # --- Kamikaze module (sacrificial explosive attack) ---
    is_kamikaze: bool = False
    kamikaze_armed: bool = False
    kamikaze_damage_hp: float = 100.0
    kamikaze_trigger_radius: float = 3.0
    
    def is_technically_available(self) -> bool:
       """Not destroyed, not EMP-disabled, not LN2-disabled."""
       return self.is_alive() and self.emp_disabled_turns <= 0 and self.ln2_disabled_turns <= 0

    def is_alive(self) -> bool:
        return self.status == DroneStatus.ACTIVE and self.hp > 0

    def is_effectively_alive(self) -> bool:
        """Alive and not currently neutralized by EMP/IEM."""
        return self.is_alive() and self.emp_disabled_turns <= 0

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
    emp_events: List[Dict[str, Any]] = field(default_factory=list)

    def init_influence_grids(self, grid_size: int = 20) -> None:
        self.influence_grid_player = [
            [0.0 for _ in range(grid_size)] for _ in range(grid_size)
        ]
        self.influence_grid_enemy = [
            [0.0 for _ in range(grid_size)] for _ in range(grid_size)
        ]

    def get_team_drones(self, team: str) -> List[DroneUnit]:
        return [d for d in self.drones if d.team == team and d.is_alive()]

    def get_team_effective_drones(self, team: str) -> List[DroneUnit]:
        return [d for d in self.drones if d.team == team and d.is_effectively_alive()]

    def is_battle_over(self) -> Optional[str]:
        """Returns:
        "player" : player wins
        "enemy"  : enemy wins
        "draw"   : both destroyed
        "timeout": max turns reached
        None     : battle continues
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
    


def compute_centroid(drones: List[DroneUnit]) -> Optional[Tuple[float, float, float]]:
    """Compute the geometric centroid (x, y, z) of a list of drones."""
    if not drones:
        return None
    sx = sum(d.x for d in drones)
    sy = sum(d.y for d in drones)
    sz = sum(d.z for d in drones)
    n = len(drones)
    return sx / n, sy / n, sz / n


@dataclass
class CampaignState:
    """Minimal campaign state compatible with the drone scenario."""
    turn: int = 0
    total_player_losses: int = 0
    total_enemy_losses: int = 0
    total_kamikaze_used: int = 0
    total_kamikaze_kills: int = 0
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
# ------------------------------------------------------------

class VolumeConstraints:
    def __init__(self):
        self.min_altitude = CONFIG["MIN_ALTITUDE"]
        # altitude ceiling per team: 1=player, 2=enemy
        self.max_altitude_team: Dict[int, float] = CONFIG["MAX_ALTITUDE_TEAM"].copy()

        # Cylindrical NFZ zones
        self.no_fly_zones: List[Dict[str, Any]] = []
        for cx, cy, radius, z_min, z_max in CONFIG["NFZ_LIST"]:
            self.add_nfz_cylinder(center=(cx, cy), radius=radius, z_min=z_min, z_max=z_max)

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
    # map team string -> index (1 or 2)
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

    # normalized
    drone.battery_level -= cons * dt
    drone.battery_level = max(drone.battery_level, 0.0)

    # Maintain compatibility with the "battery" field
    drone.battery -= cons * dt * 1000.0
    drone.battery = max(drone.battery, 0.0)

    if drone.battery <= 0 and drone.status == DroneStatus.ACTIVE:
        drone.status = DroneStatus.OUT_OF_BATTERY

    # Gradual performance reduction if battery is low
    if drone.battery_level < 0.2:
        drone.dyn_max_speed = max(5.0, drone.dyn_max_speed * 0.99)
        drone.dyn_max_accel = max(2.0, drone.dyn_max_accel * 0.99)


# ============================================================
# 3. Drone dynamics (inertia, acceleration, yaw)
# ============================================================
# ------------------------------------------------------------

def update_drone_dynamics(
    drone: DroneUnit,
    desired_vel: Tuple[float, float, float],
    dt: float,
):
    """desired_vel: Target velocity vector (vx, vy, vz) coming from the AI.
    A simple dynamic model is applied with limited acceleration and maximum
    angular velocity (yaw_rate).
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
    desired_vz = max(
        -drone.dyn_max_speed * drone.climb_speed_factor,
        min(drone.dyn_max_speed * drone.climb_speed_factor, dvz),
    )
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
# ------------------------------------------------------------

def drone_intercepted(
    drone: DroneUnit,
    campaign_state: Optional[CampaignState] = None
) -> None:
    """Handle interception: drone is shot down, falls to the ground and
    becomes 'INTERCEPTED'.
    """
    drone.hp = 0.0
    drone.status = DroneStatus.INTERCEPTED
    drone.z = 0.0  # on the ground
    drone.vx = drone.vy = drone.vz = 0.0
    
    if campaign_state is not None:
        if drone.team == "player":
            campaign_state.total_player_losses += 1
        else:
            campaign_state.total_enemy_losses += 1

    if campaign_state is not None:
        campaign_state.log(
            f"[Drone] {drone.id} intercepted and crashed to the ground"
        )


# ============================================================
# 5. EnhancedEnemyAI (with simple learning)
# ============================================================
# ------------------------------------------------------------

class EnhancedEnemyAI:
    def __init__(self, coordinator: Optional["EnemySquadCoordinator"] = None):
        self.personality: str = "deceptive"  # "aggressive" / "defensive" / "deceptive"
        self.memory: List[Dict[str, Any]] = []
        self.memory_size: int = 5

        self.coordinator = coordinator

        # learning knobs
        self.virtual_weapon_range_factor: float = 1.0
        self.aggressiveness_bias: float = 0.0
        self.preferred_distance_factor: float = 1.0

    def observe_outcome(self, outcome: Dict[str, Any]) -> None:
        """Store outcome and update coarse-grain parameters:
        - if enemy wins quickly: increase aggressiveness and virtual weapon range
        - if enemy loses: become more conservative (larger preferred distance, smaller range)
        """
        self.memory.append(outcome)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        winner = outcome.get("winner_team")
        turns = outcome.get("turns", 0)

        if winner == "enemy":
            # successful pattern: push aggressiveness & range slightly up
            speed_factor = 1.0 if turns == 0 else max(
                0.5, min(1.5, 200.0 / (turns + 1))
            )
            self.virtual_weapon_range_factor = min(
                1.6, self.virtual_weapon_range_factor + 0.05 * speed_factor
            )
            self.aggressiveness_bias = min(
                0.5, self.aggressiveness_bias + 0.02 * speed_factor
            )
            self.preferred_distance_factor = max(
                0.6, self.preferred_distance_factor - 0.02 * speed_factor
            )

        elif winner == "player":
            # bad outcome: make enemy more cautious
            self.virtual_weapon_range_factor = max(
                0.8, self.virtual_weapon_range_factor - 0.05
            )
            self.aggressiveness_bias = max(-0.3, self.aggressiveness_bias - 0.03)
            self.preferred_distance_factor = min(
                1.4, self.preferred_distance_factor + 0.03
            )

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

    def _virtual_weapon_range(self, drone: DroneUnit) -> float:
        return drone.weapon_range * max(
            0.6, min(1.6, self.virtual_weapon_range_factor)
        )

    def recommend_drone_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        if not drone.is_alive():
            return (0.0, 0.0, 0.0), None

        enemies = state.get_team_effective_drones(
            "player" if drone.team == "enemy" else "enemy"
        )
        if not enemies:
            return (0.0, 0.0, 0.0), None
        
        # Kamikaze-specific behavior
        if drone.role == DroneRole.KAMIKAZE and drone.is_kamikaze:
            return self._kamikaze_action(drone, state)

        # Focus fire: shared focus target if available
        focus_target = None
        if self.coordinator is not None and drone.team == "enemy":
            focus_target = self.coordinator.select_focus_target(state)

        target = focus_target if focus_target is not None else min(
            enemies, key=lambda e: drone.distance_to(e)
        )
        dist = drone.distance_to(target)

        # Personality-based base velocity
        if self.personality == "aggressive":
            vx, vy, vz = self._vector_towards(
                drone, target, factor=1.0 + self.aggressiveness_bias
            )
            shoot = target if (
                dist <= self._virtual_weapon_range(drone) and drone.ammo > 0
            ) else None

        elif self.personality == "defensive":
            desired = (
                self._virtual_weapon_range(drone)
                * 0.7
                * self.preferred_distance_factor
            )
            if dist < desired:
                vx, vy, vz = self._vector_away(drone, target, factor=1.0)
            else:
                vx, vy, vz = self._vector_towards(
                    drone, target, factor=0.5 + self.aggressiveness_bias
                )
            shoot = target if (
                dist <= self._virtual_weapon_range(drone) * 0.9
                and drone.ammo > 0
            ) else None

        else:
            # deceptive
            if drone.hp < 40:
                vx, vy, vz = self._vector_away(drone, target, factor=1.0)
            else:
                vx, vy, vz = self._flank_vector(drone, target)
            shoot = target if (
                dist <= self._virtual_weapon_range(drone) * 0.8
                and drone.ammo > 0
            ) else None

        # Encirclement tweak: add lateral component suggested by coordinator
        if self.coordinator is not None and drone.team == "enemy":
            ex, ey = self.coordinator.encirclement_direction(drone, target)
            vx += 0.4 * ex * drone.dyn_max_speed
            vy += 0.4 * ey * drone.dyn_max_speed

        return (vx, vy, vz), shoot

    def _kamikaze_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        enemies = state.get_team_effective_drones(
            "player" if drone.team == "enemy" else "enemy"
        )
        target = min(enemies, key=lambda e: drone.distance_to(e))
        dist = drone.distance_to(target)

        low_own = (drone.hp < CONFIG["LOW_HP_THRESHOLD"]) or (drone.battery_level < 0.2)
        high_value_target = (
            target.role in (DroneRole.SNIPER, DroneRole.TANK)
            or target.hp > 40.0
        )

        # personality modulates how easily we accept a sacrificial trade
        if self.personality == "aggressive":
            threshold = 0.6
        elif self.personality == "defensive":
            threshold = 1.2
        else:  # deceptive
            threshold = 0.9

        # base decision: attack if we are weak OR target is very high value
        decision_score = 0.0
        if low_own:
            decision_score += 1.0
        if high_value_target:
            decision_score += 1.0

        if decision_score >= threshold:
            vx, vy, vz = self._vector_towards(drone, target, factor=1.0)
        else:
            vx, vy, vz = self._flank_vector(drone, target)

        shoot = None
        return (vx, vy, vz), shoot


    def _vector_towards(
        self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0
    ):
        dx, dy, dz = (
            target.x - drone.x,
            target.y - drone.y,
            target.z - drone.z,
        )
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _vector_away(
        self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0
    ):
        dx, dy, dz = (
            drone.x - target.x,
            drone.y - target.y,
            drone.z - target.z,
        )
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
# ------------------------------------------------------------

class ChessSunTzuAI:
    def __init__(self, map_width: int = 100, map_height: int = 100):
        self.map_width = map_width
        self.map_height = map_height

    def positional_score_for_team(
        self, state: DroneScenarioState, team: str
    ) -> float:
        drones = state.get_team_drones(team)
        if not drones:
            return -999.0

        score = 0.0
        cx, cy = self.map_width / 2.0, self.map_height / 2.0

        for d in drones:
            dist_center = math.hypot(d.x - cx, d.y - cy)
            score += max(0.0, 50.0 - dist_center)
            score += d.hp * 0.5
            margin = min(
                d.x, d.y, self.map_width - d.x, self.map_height - d.y
            )
            score += max(-30.0, margin - 30.0)

        return score


class GoSunTzuAI:
    def __init__(
        self,
        grid_size: int = 20,
        map_width: int = 100,
        map_height: int = 100
    ):
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
                        infl = math.exp(-(dist ** 2) / (2 * sigma ** 2)) * (
                            1.0 + d.hp / 100.0
                        )
                        if d.team == "player":
                            player_infl += infl
                        else:
                            enemy_infl += infl

                state.influence_grid_player[i][j] = player_infl
                state.influence_grid_enemy[i][j] = enemy_infl

    def detect_encirclement_zones(
        self, state: DroneScenarioState, target_team: str = "enemy"
    ) -> List[DroneUnit]:
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
# 7. PlayerDroneController (tactical, player side)
# ============================================================
# ------------------------------------------------------------

class PlayerDroneController:
    def __init__(
        self,
        controls: Dict[str, Any],
        commander: Optional["SquadronCommander"] = None
    ):
        self.controls = controls
        self.commander = commander

    def _get_personality(self) -> str:
        return self.controls.get("player_personality", "defensive")

    def _get_aggressiveness(self) -> float:
        return float(self.controls.get("player_aggressiveness", 0.5))

    def _get_pref_distance(self, default: float) -> float:
        return float(self.controls.get("player_pref_distance", default))

    def recommend_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        enemies = state.get_team_effective_drones("enemy")
        if not drone.is_alive() or not enemies:
            return (0.0, 0.0, 0.0), None
        
        # Kamikaze-specific behavior (sacrificial attack)
        if drone.role == DroneRole.KAMIKAZE and drone.is_kamikaze:
            return self._kamikaze_action(drone, state)

        target = min(enemies, key=lambda e: drone.distance_to(e))
        dist = drone.distance_to(target)

        personality = self._get_personality()
        aggr = self._get_aggressiveness()
        pref_dist = self._get_pref_distance(drone.weapon_range * 0.8)

        # Role-aware tuning from SquadronCommander
        if self.commander is not None:
            rp = self.commander.get_role_params(drone)
            aggr *= rp.get("aggressiveness", 1.0)
            pref_dist *= rp.get("pref_factor", 1.0)

        if personality == "aggressive":
            factor = 0.5 + aggr * 0.5
            vx, vy, vz = self._vector_towards(drone, target, factor=factor)
            shoot = target if (dist <= drone.weapon_range and drone.ammo > 0) else None

        elif personality == "defensive":
            if dist < pref_dist:
                vx, vy, vz = self._vector_away(
                    drone, target, factor=0.5 + (1.0 - aggr) * 0.5
                )
            else:
                vx, vy, vz = self._vector_towards(
                    drone, target, factor=0.3 + aggr * 0.2
                )
            shoot = target if (
                dist <= drone.weapon_range * 0.9 and drone.ammo > 0
            ) else None

        else:
            # deceptive
            if drone.hp < 40:
                vx, vy, vz = self._vector_away(
                    drone, target, factor=0.7 + (1.0 - aggr) * 0.3
                )
            else:
                vx, vy, vz = self._flank_vector(drone, target)
            shoot = target if (
                dist <= drone.weapon_range * 0.8 and drone.ammo > 0
            ) else None

        return (vx, vy, vz), shoot
    
    def _kamikaze_action(
        self, drone: DroneUnit, state: DroneScenarioState
    ) -> Tuple[Tuple[float, float, float], Optional[DroneUnit]]:
        """
        Sun Tzu-inspired kamikaze logic:
        - If the drone is already heavily damaged or low on battery,
          it accepts an unfavorable exchange to remove a high-value target.
        - Otherwise, it approaches and flanks, waiting for a better moment.
        """
        enemies = state.get_team_effective_drones("enemy")
        target = min(enemies, key=lambda e: drone.distance_to(e))
        dist = drone.distance_to(target)

        # own weakness: when we are already "lost", we become very aggressive
        low_own = (drone.hp < CONFIG["LOW_HP_THRESHOLD"]) or (drone.battery_level < 0.2)

        # high-value target: SNIPER / TANK or high HP
        high_value_target = (
            target.role in (DroneRole.SNIPER, DroneRole.TANK)
            or target.hp > 40.0
        )

        # Optional: check local influence to avoid wasting kamikaze
        go_ai = None
        if self.commander is not None:
            # commander may expose go_ai later, kept simple for now
            pass

        if low_own and high_value_target:
            # "If you must perish, strike at the enemy's strength"
            vx, vy, vz = self._vector_towards(drone, target, factor=1.0)
        else:
            # wait for better trade: flanking, probing defenses
            vx, vy, vz = self._flank_vector(drone, target)

        # no standard shooting: the body is the weapon
        shoot = None
        return (vx, vy, vz), shoot
    

    def _vector_towards(
        self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0
    ):
        dx, dy, dz = (
            target.x - drone.x,
            target.y - drone.y,
            target.z - drone.z,
        )
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.dyn_max_speed,
            factor * dy / norm * drone.dyn_max_speed,
            factor * dz / norm * drone.dyn_max_speed * drone.climb_speed_factor,
        )

    def _vector_away(
        self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0
    ):
        dx, dy, dz = (
            drone.x - target.x,
            drone.y - target.y,
            drone.z - target.z,
        )
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
# 8. SquadronCommander (strategic per-team role manager)
# ============================================================
# ------------------------------------------------------------

class SquadronCommander:
    """Strategic layer per team.
    - Assigns roles (SCOUT, SNIPER, TANK, DECOY)
    - Provides role-aware parameters for lower-level controllers.
    """

    def __init__(self, team: str):
        self.team = team

        # virtual tuning knobs per role
        self.role_params: Dict[DroneRole, Dict[str, float]] = {
            DroneRole.SCOUT: {"aggressiveness": 0.7, "pref_factor": 1.2},
            DroneRole.SNIPER: {"aggressiveness": 0.5, "pref_factor": 1.6},
            DroneRole.TANK: {"aggressiveness": 0.9, "pref_factor": 0.6},
            DroneRole.DECOY: {"aggressiveness": 0.8, "pref_factor": 1.0},
        }

    def assign_roles(self, drones: List[DroneUnit]) -> None:
        team_drones = [d for d in drones if d.team == self.team]
        if not team_drones:
            return

        n = len(team_drones)
        # simple heuristic: first = TANK, last = SCOUT, middle = SNIPER/DECOY
        for i, d in enumerate(sorted(team_drones, key=lambda x: x.id)):
            if i == 0:
                d.role = DroneRole.TANK
            elif i == n - 1:
                d.role = DroneRole.SCOUT
            elif i % 2 == 0:
                d.role = DroneRole.SNIPER
            else:
                d.role = DroneRole.DECOY

    def get_role_params(self, drone: DroneUnit) -> Dict[str, float]:
        return self.role_params.get(drone.role, self.role_params[DroneRole.SCOUT])

# ============================================================
# 9. EnemySquadCoordinator (focus fire & encirclement)
# ============================================================
# ------------------------------------------------------------

class EnemySquadCoordinator:
    """Coordinates enemy drones:
    - selects a primary focus target among player drones
    - suggests encirclement directions using GoSunTzuAI influence
    - computes safe arc positions for EMP tactics
    """

    def __init__(self, go_ai: GoSunTzuAI):
        self.go_ai = go_ai

    def select_focus_target(
        self, state: DroneScenarioState
    ) -> Optional[DroneUnit]:
        players = state.get_team_effective_drones("player")
        if not players:
            return None

        # prioritize low HP + close to center (easy to finish)
        cx, cy = state.map_width / 2.0, state.map_height / 2.0

        def score(d: DroneUnit) -> float:
            return d.hp + 0.5 * math.hypot(d.x - cx, d.y - cy)

        return min(players, key=score)

    def encirclement_direction(
        self, enemy: DroneUnit, target: DroneUnit
    ) -> Tuple[float, float]:
        """Returns a lateral direction around the target to encourage encirclement.
        Simple: perpendicular vector, oriented by enemy index relative to target.
        """
        dx, dy = target.x - enemy.x, target.y - enemy.y
        norm = math.hypot(dx, dy) + 1e-6
        fx, fy = -dy / norm, dx / norm
        # pick side deterministically from IDs
        sign = 1.0 if enemy.id < target.id else -1.0
        return sign * fx, sign * fy

    # ---------- ARC TACTICS FOR EMP ----------

    def arc_positions_around_cluster(
        self,
        center: Tuple[float, float],
        radius: float,
        n_points: int,
        facing_from: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """
        Generate n_points positions on an arc of circle centered on 'center',
        with radius 'radius', oriented roughly in the direction 'facing_from'.
        """
        cx, cy = center
        fx = facing_from[0] - cx
        fy = facing_from[1] - cy
        base_angle = math.atan2(fy, fx)

        arc_angle = math.radians(120.0)
        start_angle = base_angle - arc_angle / 2.0
        step = arc_angle / max(1, n_points - 1)

        positions = []
        for k in range(n_points):
            ang = start_angle + k * step
            x = cx + radius * math.cos(ang)
            y = cy + radius * math.sin(ang)
            positions.append((x, y))
        return positions

    def compute_emp_arc_assignments(
        self,
        state: DroneScenarioState,
        shooter: DroneUnit,
        enemy_cluster: List[DroneUnit],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute safe arc positions around an enemy cluster for a subset of allies.
        Returns: { drone_id: (x_arc, y_arc) }.
        """
        if not enemy_cluster:
            return {}

        emp_range = CONFIG["EMP_RANGE"]
        safe_radius = emp_range * CONFIG["EMP_SAFE_FACTOR"]

        centroid = compute_centroid(enemy_cluster)
        if centroid is None:
            return {}
        cx, cy, cz = centroid

        allies = [
            d for d in state.drones
            if d.team == shooter.team and d.is_alive()
        ]
        allies_sorted = sorted(
            allies, key=lambda d: math.hypot(d.x - cx, d.y - cy)
        )
        max_arc = CONFIG["EMP_MAX_ARC_DRONES"]
        selected = allies_sorted[:max_arc]

        arc_positions = self.arc_positions_around_cluster(
            center=(cx, cy),
            radius=safe_radius,
            n_points=len(selected),
            facing_from=(shooter.x, shooter.y),
        )

        assignments: Dict[str, Tuple[float, float]] = {}
        for d, pos in zip(selected, arc_positions):
            assignments[d.id] = pos
        return assignments



# ============================================================
# 10. Drone initialization helper (variable counts)
# ============================================================
# ------------------------------------------------------------

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

    # Tag EMP-capable drones (SPEAR / THOR) if feature enabled
    if CONFIG.get("ENABLE_EMP_WEAPONS", False):
        for d in drones:
            if d.team == "player":
                if random.random() < CONFIG["PLAYER_EMP_PROB"]:
                    d.has_emp = True
            else:
                if random.random() < CONFIG["ENEMY_EMP_PROB"]:
                    d.has_emp = True
                    
    # Tag LN2-capable drones (liquid nitrogen spray) if feature enabled
    if CONFIG.get("ENABLE_LN2_WEAPON", False):
        for d in drones:
            if d.team == "player":
                if random.random() < CONFIG["LN2_PROB_PLAYER"]:
                    d.has_ln2 = True
            else:
                if random.random() < CONFIG["LN2_PROB_ENEMY"]:
                    d.has_ln2 = True
                    
    # Tag Kamikaze-capable drones (sacrificial attack)  if feature enabled
    if CONFIG.get("ENABLE_KAMIKAZE_WEAPON", False):
        for d in drones:
            if d.team == "player":
                if random.random() < CONFIG["KAMIKAZE_PROB_PLAYER"]:
                    d.is_kamikaze = True
            else:
                if random.random() < CONFIG["KAMIKAZE_PROB_ENEMY"]:
                    d.is_kamikaze = True
    
            if d.is_kamikaze:
                d.role = DroneRole.KAMIKAZE
                d.kamikaze_armed = True
                d.kamikaze_damage_hp = CONFIG["KAMIKAZE_DAMAGE_HP"]
                d.kamikaze_trigger_radius = CONFIG["KAMIKAZE_TRIGGER_RADIUS"]

    return drones


# ============================================================
# 11. Collision avoidance helper
# ============================================================
# ------------------------------------------------------------

def apply_collision_avoidance(
    ds: DroneScenarioState,
    min_dist: float = 3.0,
    strength: float = 1.5
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
# 11bis. EMP / IEM helpers
# ============================================================
# ------------------------------------------------------------

def update_emp_status(drone: DroneUnit) -> None:
    """Update EMP cooldown and disable duration."""
    if drone.emp_cooldown > 0:
        drone.emp_cooldown -= 1
    if drone.emp_disabled_turns > 0:
        drone.emp_disabled_turns -= 1

def update_ln2_status(drone: DroneUnit) -> None:
    """Update LN2 cooldown and disable duration."""
    if drone.ln2_cooldown > 0:
        drone.ln2_cooldown -= 1
    if drone.ln2_disabled_turns > 0:
        drone.ln2_disabled_turns -= 1

def decide_emp_use(
    shooter: DroneUnit,
    state: DroneScenarioState,
    coordinator: Optional["EnemySquadCoordinator"] = None,
) -> List[DroneUnit]:
    if not CONFIG.get("ENABLE_EMP_WEAPONS", False):
        return []
    if not getattr(shooter, "has_emp", False):
        return []
    if shooter.emp_cooldown > 0 or shooter.emp_disabled_turns > 0:
        return []

    emp_range = CONFIG["EMP_RANGE"]
    min_enemies = CONFIG["EMP_MIN_ENEMIES"]
    safe_radius = emp_range * CONFIG["EMP_SAFE_FACTOR"]

    enemies = state.get_team_drones(
        "player" if shooter.team == "enemy" else "enemy"
    )

    in_range = [
        e for e in enemies
        if e.is_alive() and shooter.distance_to(e) <= emp_range
    ]
    if len(in_range) < min_enemies:
        return []

    # Allied security
    allies = [
        d for d in state.drones
        if d.team == shooter.team and d.is_alive()
    ]
    for ally in allies:
        if ally.id == shooter.id:
            continue
        if shooter.distance_to(ally) <= safe_radius:
            return []

    # Option: prepare the arc positions (for later use)
    if coordinator is not None:
        _assignments = coordinator.compute_emp_arc_assignments(
            state=state,
            shooter=shooter,
            enemy_cluster=in_range,
        )

    return in_range

def decide_ln2_use(
    shooter: DroneUnit,
    state: "DroneScenarioState",
) -> List[DroneUnit]:
    """
    Decide whether to use the liquid nitrogen spray (short-range conical weapon).
    Returns the list of enemy drones inside the cone.
    """
    if not CONFIG.get("ENABLE_LN2_WEAPON", False):
        return []
    if not getattr(shooter, "has_ln2", False):
        return []
    if shooter.ln2_cooldown > 0 or shooter.emp_disabled_turns > 0 or shooter.ln2_disabled_turns > 0:
        return []

    ln2_range = CONFIG["LN2_RANGE"]
    cone_angle_deg = CONFIG["LN2_CONE_ANGLE_DEG"]
    cos_max = math.cos(math.radians(cone_angle_deg))

    # enemies in front of the drone (cone from yaw direction)
    enemies = state.get_team_drones("player" if shooter.team == "enemy" else "enemy")
    impacted: List[DroneUnit] = []

    # heading vector in XY from shooter.yaw
    hx = math.cos(shooter.yaw)
    hy = math.sin(shooter.yaw)

    for e in enemies:
        if not e.is_alive():
            continue
        dx = e.x - shooter.x
        dy = e.y - shooter.y
        dz = e.z - shooter.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > ln2_range:
            continue

        # angle between heading (hx, hy) and target vector projected in XY
        norm_xy = math.sqrt(dx * dx + dy * dy)
        if norm_xy < 1e-6:
            continue
        tx = dx / norm_xy
        ty = dy / norm_xy
        cos_angle = hx * tx + hy * ty
        if cos_angle >= cos_max:
            impacted.append(e)

    return impacted

def apply_ln2_effects(
    shooter: DroneUnit,
    targets: List[DroneUnit],
    state: "DroneScenarioState",
    campaign_state: CampaignState,  
) -> None:
    """Apply mechanical/thermal effects of liquid nitrogen spray."""
    if not targets:
        return

    dmg_hp = CONFIG["LN2_DAMAGE_HP"]
    disable_turns = CONFIG["LN2_DISABLE_TURNS"]
    battery_penalty = CONFIG["LN2_BATTERY_PENALTY"]

    for t in targets:
        if not t.is_alive():
            continue
        t.hp = max(0.0, t.hp - dmg_hp)
        t.battery = max(0.0, t.battery - battery_penalty)
        t.ln2_disabled_turns = max(t.ln2_disabled_turns, disable_turns)

        if t.hp <= 0 and t.status == DroneStatus.ACTIVE:
            drone_intercepted(t, campaign_state)  # on log l’interception

    shooter.ln2_cooldown = CONFIG["LN2_COOLDOWN_TURNS"]

    # use emp_events for logger LN2 (simple)
    state.emp_events.append(
        {
            "turn": state.turn_index,
            "shooter_id": shooter.id,
            "shooter_team": shooter.team,
            "num_targets": len(targets),
            "weapon": "LN2",
        }
    )


# ============================================================
# 11ter. Kamikaze attack helper
# ============================================================
# ------------------------------------------------------------

def try_kamikaze_attack(state: DroneScenarioState,
                        campaign_state: Optional[CampaignState] = None) -> None:
    """
    Check all kamikaze drones and trigger an explosion if an enemy
    enters the trigger radius. Sacrificial, Sun Tzu-style exchange:
    one unit for a high-value target.
    """
    for d in state.drones:
        if not d.is_alive() or not d.is_kamikaze or not d.kamikaze_armed:
            continue

        enemies = state.get_team_effective_drones(
            "enemy" if d.team == "player" else "player"
        )
        if not enemies:
            continue

        # nearest enemy
        target = min(enemies, key=lambda e: d.distance_to(e))
        dist = d.distance_to(target)

        if dist <= d.kamikaze_trigger_radius:
            # explosion damage
            target.hp -= d.kamikaze_damage_hp
            if target.hp <= 0 and target.status == DroneStatus.ACTIVE:
                drone_intercepted(target, campaign_state)

            # kamikaze drone is destroyed
            d.hp = 0.0
            d.status = DroneStatus.DESTROYED
            d.kamikaze_armed = False
            d.vx = d.vy = d.vz = 0.0
            
            if campaign_state is not None:
                campaign_state.total_kamikaze_used += 1
                if target.hp <= 0:
                    campaign_state.total_kamikaze_kills += 1

            if campaign_state is not None:
                campaign_state.log(
                    f"[KAMIKAZE] {d.id} explodes on {target.id}, "
                    f"damage={d.kamikaze_damage_hp:.1f}, dist={dist:.2f}"
                )



# ============================================================
# 12. Export shooting/battery/interception stats
# ============================================================
# ------------------------------------------------------------

def export_battle_stats(ds: DroneScenarioState, campaign_state: CampaignState) -> None:
    distances_player = [
        e["distance"] for e in ds.shot_events if e["shooter_team"] == "player"
    ]
    distances_enemy = [
        e["distance"] for e in ds.shot_events if e["shooter_team"] == "enemy"
    ]

    hits_player = [
        e for e in ds.shot_events
        if e["shooter_team"] == "player" and e["hit"]
    ]
    hits_enemy = [
        e for e in ds.shot_events
        if e["shooter_team"] == "enemy" and e["hit"]
    ]

    total_player = len(
        [e for e in ds.shot_events if e["shooter_team"] == "player"]
    )
    total_enemy = len(
        [e for e in ds.shot_events if e["shooter_team"] == "enemy"]
    )

    hit_rate_player = len(hits_player) / total_player if total_player > 0 else 0.0
    hit_rate_enemy = len(hits_enemy) / total_enemy if total_enemy > 0 else 0.0

    # Average battery life (last snapshot)
    if ds.history:
        last_snap = ds.history[-1]
        final_batteries_player = [
            last_snap[did][4]
            for did in last_snap.keys()
            if did.startswith("B")
        ]
        final_batteries_enemy = [
            last_snap[did][4]
            for did in last_snap.keys()
            if did.startswith("R")
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

    interception_turns_player: List[int] = []
    interception_turns_enemy: List[int] = []
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
        f"[Stats] Player hit rate={hit_rate_player:.2f}, "
        f"Enemy hit rate={hit_rate_enemy:.2f}"
    )
    campaign_state.log(
        f"[Stats] Avg battery P={avg_batt_player:.1f}, "
        f"E={avg_batt_enemy:.1f}"
    )
    campaign_state.log(
        f"[Stats] Avg turns before intercept P={avg_turn_intercept_player:.1f}, "
        f"E={avg_turn_intercept_enemy:.1f}"
    )

    # CSV export: kinetic shots
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
        campaign_state.log(
            f"[Stats] Could not write shot_events.csv: {ex}"
        )

    # CSV export: EMP events
    try:
        with open("emp_events.csv", "w", encoding="utf-8") as f:
            f.write("turn,shooter_id,team,num_targets\n")
            for e in ds.emp_events:
                f.write(
                    f"{e['turn']},{e['shooter_id']},{e['shooter_team']},"
                    f"{e['num_targets']}\n"
                )
    except Exception as ex:
        campaign_state.log(
            f"[Stats] Could not write emp_events.csv: {ex}"
        )

# ============================================================
# 13. Main loop: run_drone_campaign
# ============================================================
# ------------------------------------------------------------

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

    chess_ai = ChessSunTzuAI(
        map_width=ds.map_width, map_height=ds.map_height
    )
    go_ai = GoSunTzuAI(
        grid_size=50, map_width=ds.map_width, map_height=ds.map_height
    )

    # Strategic commanders
    enemy_commander = SquadronCommander(team="enemy")
    player_commander = SquadronCommander(team="player")
    enemy_commander.assign_roles(ds.drones)
    player_commander.assign_roles(ds.drones)

    # Enemy coordination
    coordinator = EnemySquadCoordinator(go_ai)

    # Enemy AI (may persist across battles)
    enemy_ai: EnhancedEnemyAI = campaign_state.enemy_ai
    if enemy_ai is None:
        enemy_ai = EnhancedEnemyAI(coordinator=coordinator)
        campaign_state.enemy_ai = enemy_ai
    else:
        enemy_ai.coordinator = coordinator

    player_ai = PlayerDroneController(
        campaign_state.player_controls,
        commander=player_commander,
    )

    # Volume & Weather
    constraints = VolumeConstraints()
    weather = Weather()

    preset = CONFIG["WEATHER_PRESETS"].get(
        campaign_state.weather,
        CONFIG["WEATHER_PRESETS"]["CLEAR"],
    )
    weather.wind_vector = np.array(preset["wind"], dtype=float)
    weather.turbulence_std_xy = preset["turb_xy"]
    weather.turbulence_std_z = preset["turb_z"]
    weather.battery_drain_factor = preset["battery_factor"]

    campaign_state.log(
        f"=== Drone Campaign: {num_red} enemy drones vs {num_blue} player drones "
        f"(EMP enabled={CONFIG.get('ENABLE_EMP_WEAPONS', False)}) ==="
    )

    dt = 1.0  # time step

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
        encircled_player = go_ai.detect_encirclement_zones(
            ds, target_team="player"
        )

        campaign_state.log(
            f"[DroneChess] enemy_pos={chess_enemy:.1f}, "
            f"player_pos={chess_player:.1f}"
        )
        
        if encircled_player:
            names = ", ".join(d.id for d in encircled_player)
            campaign_state.log(
                f"[DroneGo] Player drones encircled: {names}"
            )

        # ---- AI recommendations (movement + potential kinetic shot) ----
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

        # Collision avoidance
        extra_vel = apply_collision_avoidance(
            ds, min_dist=3.0, strength=1.5
        )

        # ---- Movement + dynamics + weather + volume + battery + EMP status ----
        for d in ds.drones:
            if not d.is_alive():
                continue

            # Drone neutralized by EMP/IEM: no movement or AI this turn
            if getattr(d, "emp_disabled_turns", 0) > 0:
                update_emp_status(d)
                update_ln2_status(d)
                update_battery(d, weather, dt)
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
            update_emp_status(d)
            update_ln2_status(d)

        # ---- EMP / IEM firing phase (SPEAR / THOR) ----
        if CONFIG.get("ENABLE_EMP_WEAPONS", False):
            for shooter in ds.drones:
                if not shooter.is_alive():
                    continue

                targets_emp = decide_emp_use(shooter, ds, coordinator)
                if not targets_emp:
                    continue

                for target in targets_emp:
                    target.emp_disabled_turns = max(
                        getattr(target, "emp_disabled_turns", 0),
                        CONFIG["EMP_DISABLE_DURATION"],
                    )
                    target.hp -= CONFIG["EMP_DAMAGE_HP"]
                    target.battery = max(
                        0.0,
                        target.battery - CONFIG["EMP_DAMAGE_BATTERY"],
                    )
                    if (
                        target.hp <= 0
                        and target.status == DroneStatus.ACTIVE
                    ):
                        drone_intercepted(target, campaign_state)

                shooter.emp_cooldown = CONFIG["EMP_COOLDOWN_TURNS"]

                side = "BLUE" if shooter.team == "player" else "RED"
                weapon_name = "SPEAR" if shooter.team == "player" else "THOR"
                campaign_state.log(
                    f"[EMP-{side}] {shooter.id} activates {weapon_name} IEM on "
                    f"{len(targets_emp)} target(s) (r={CONFIG['EMP_RANGE']:.1f})"
                )

                ds.emp_events.append(
                    {
                        "turn": ds.turn_index,
                        "shooter_id": shooter.id,
                        "shooter_team": shooter.team,
                        "num_targets": len(targets_emp),
                    }
                )
                
        # >>> LN2 : new phase between EMP and kinetic strike
        # ---- LN2 (liquid nitrogen) firing phase ----
        if CONFIG.get("ENABLE_LN2_WEAPON", False):
            for shooter in ds.drones:
                if not shooter.is_alive():
                    continue
                if getattr(shooter, "ln2_disabled_turns", 0) > 0:
                    continue
                if not getattr(shooter, "has_ln2", False):
                    continue

                targets_ln2 = decide_ln2_use(shooter, ds)
                if not targets_ln2:
                    continue

                apply_ln2_effects(shooter, targets_ln2, ds, campaign_state)

                side = "BLUE" if shooter.team == "player" else "RED"
                campaign_state.log(
                    f"[LN2-{side}] {shooter.id} sprays liquid nitrogen on "
                    f"{len(targets_ln2)} target(s) (r={CONFIG['LN2_RANGE']:.1f}, "
                    f"cone={CONFIG['LN2_CONE_ANGLE_DEG']:.1f}°)"
                )

        # ---- Kinetic firing phase + stats ----
        for shooter, target in planned_shots:
            if not shooter.is_alive() or not target.is_alive():
                continue
            if getattr(shooter, "emp_disabled_turns", 0) > 0:
                continue
            if getattr(shooter, "ln2_disabled_turns", 0) > 0:  # >>> LN2 : Shooting is blocked if the drone is frozen. ;-)
                continue
            if shooter.ammo <= 0:
                continue

            dist = shooter.distance_to(target)
            if dist > shooter.weapon_range:
                continue

            hit_prob = max(
                0.5,
                1.5 / (shooter.weapon_range + 1e-6) * shooter.weapon_range
                - dist / (shooter.weapon_range + 1e-6),
            )

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
                    f"[Drone] {shooter.id} hits {target.id} for "
                    f"{dmg:.1f} HP (dist={dist:.1f})"
                )
                if target.hp <= 0 and target.status == DroneStatus.ACTIVE:
                    drone_intercepted(target, campaign_state)
            else:
                campaign_state.log(
                    f"[Drone] {shooter.id} misses {target.id} "
                    f"(dist={dist:.1f})"
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
          
        if CONFIG.get("ENABLE_KAMIKAZE_WEAPON", False):
            try_kamikaze_attack(ds, campaign_state)

        # ---- History snapshot ----
        snapshot: Dict[str, Tuple[float, float, float, float, float, DroneStatus]] = {}
        for d in ds.drones:
            snapshot[d.id] = (
                d.x,
                d.y,
                d.z,
                d.hp,
                d.battery,
                d.status,
            )
        ds.history.append(snapshot)


    metrics = compute_sun_tzu_metrics(campaign_state)
    sun_tzu_score = compute_sun_tzu_score(metrics, victory=True)

    campaign_state.log(
        f"[SUN-TZU] exchange={metrics['exchange_ratio']:.2f}, "
        f"kamikaze_eff={metrics['kamikaze_efficiency']:.2f}, "
        f"cost={metrics['cost_of_victory']:.1f}, "
        f"score={sun_tzu_score:.1f}"
    )


    export_battle_stats(ds, campaign_state)
    campaign_state.drone_scenario = ds
    

    


# ============================================================
# 14. Visualisations
# ============================================================
# ------------------------------------------------------------

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

    # Index EMP events by turn (ds.emp_events contient déjà les tirs EMP)
    emp_by_turn: Dict[int, Dict[str, bool]] = {}
    for e in ds.emp_events:
        t = e["turn"]         # same convention as shot_events: 1..N
        team = e["shooter_team"]  # "player" ou "enemy"
        if t not in emp_by_turn:
            emp_by_turn[t] = {"player": False, "enemy": False}
        emp_by_turn[t][team] = True

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    player_high = ax.scatter([], [], [], c="blue", label="Player (>LOW_HP)")
    player_low = ax.scatter([], [], [], c="cyan", label="Player (<=LOW_HP)")
    enemy_high = ax.scatter([], [], [], c="red", label="Enemy (>LOW_HP)")
    enemy_low = ax.scatter([], [], [], c="orange", label="Enemy (<=LOW_HP)")
    shot_hits = ax.scatter([], [], [], c="green", marker="*", s=50, label="Hit")
    shot_misses = ax.scatter([], [], [], c="yellow", marker="x", s=40, label="Miss")
    intercepted_scatter = ax.scatter([], [], [], c="black", label="Intercepted")

    #ax.set_xlim(0, ds.map_width)
    #ax.set_ylim(0, ds.map_height)
    
    ax.set_xlim(-ds.map_width, ds.map_width)
    ax.set_ylim(-ds.map_height, ds.map_height) 
    
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

        # Shots in this round (turn_index starts at 1)
        turn_shots = [
            e for e in ds.shot_events if e["turn"] == frame + 1
        ]
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

        # EMP info for this frame (events are stockés avec 'turn' 1..N)
        emp_info = ""
        turn_emp = emp_by_turn.get(frame + 1, None)
        if turn_emp:
            flags = []
            if turn_emp.get("player"):
                flags.append("EMP BLUE")
            if turn_emp.get("enemy"):
                flags.append("EMP RED")
            if flags:
                emp_info = " | " + " & ".join(flags)

        info_text.set_text(
            f"Turn {frame+1}/{turns}  "
            f"Alive={alive_count}  "
            f"Red intercepted={red_intercepted}  "
            f"Blue intercepted={blue_intercepted}{emp_info}"
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
    #anim = FuncAnimation(fig, update, frames=turns, interval=100, blit=False)
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
    ax_xy.set_title("Ground trajectories (X–Y)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    #ax_xy.set_xlim(0, ds.map_width)
    #ax_xy.set_ylim(0, ds.map_height)
    
    ax_xy.set_xlim(-ds.map_width , ds.map_width)
    ax_xy.set_ylim(-ds.map_height, ds.map_height)

    # altitude vs time (plot a few drones)
    max_to_plot = min(6, len(drone_ids))
    max_to_plot = len(drone_ids)
    for d_id in drone_ids[:max_to_plot]:
        t_axis = list(range(turns))
        if teams[d_id] == "player":
            ax_z.plot(t_axis, zs[d_id], label=d_id, color="blue", alpha=0.7)
        else:
            ax_z.plot(t_axis, zs[d_id], label=d_id, color="red", alpha=0.7)
    ax_z.set_title("Altitude vs time")
    ax_z.set_xlabel("Turn")
    ax_z.set_ylabel("Altitude z")
    ax_z.legend()

    plt.tight_layout()
    plt.show()


def visualize_influence_heatmaps(
    campaign_state: CampaignState,
    turns_to_show: Optional[List[int]] = None
) -> None:
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
    go_ai = GoSunTzuAI(
        grid_size=grid_size,
        map_width=ds.map_width,
        map_height=ds.map_height
    )

    fig, axes = plt.subplots(
        len(turns_to_show),
        2,
        figsize=(8, 3 * len(turns_to_show))
    )
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
        im0 = ax_p.imshow(
            np.array(ds.influence_grid_player).T,
            origin="lower",
            cmap="Blues",
        )
        im1 = ax_e.imshow(
            np.array(ds.influence_grid_enemy).T,
            origin="lower",
            cmap="Reds",
        )
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
#  Debiefing
# ============================================================
# ------------------------------------------------------------

def compute_sun_tzu_metrics(campaign: CampaignState) -> Dict[str, float]:
    if campaign.total_player_losses == 0:
        exchange_ratio = float("inf") if campaign.total_enemy_losses > 0 else 1.0
    else:
        exchange_ratio = campaign.total_enemy_losses / max(1, campaign.total_player_losses)

    kamikaze_eff = 0.0
    if campaign.total_kamikaze_used > 0:
        kamikaze_eff = campaign.total_kamikaze_kills / campaign.total_kamikaze_used

    cost_of_victory = campaign.total_player_losses - campaign.total_enemy_losses

    return {
        "exchange_ratio": exchange_ratio,
        "kamikaze_efficiency": kamikaze_eff,
        "cost_of_victory": cost_of_victory,
    }

def compute_sun_tzu_score(metrics: Dict[str, float], victory: bool) -> float:
    exchange = metrics["exchange_ratio"]
    kamikaze_eff = metrics["kamikaze_efficiency"]
    cost = metrics["cost_of_victory"]

    score = 0.0

    # Very balanced victory/defeat
    score += 50.0 if victory else -50.0

    # Sun Tzu: Win by losing little -> big bonus if exchange_ratio > 1
    score += 20.0 * (exchange - 1.0)

    # Kamikaze: bonus if the sacrifices are profitable (efficiency > 1)
    score += 15.0 * (kamikaze_eff - 1.0)

    # Penalty if the cost of victory is high (many losses)
    score -= 2.0 * max(0.0, cost)

    return score



# ============================================================
# 
# ============================================================
# ------------------------------------------------------------

if __name__ == "__main__":

    # ----- Global Test Settings -----
    # Enable or disable SPEAR/THOR EMP weapons or KAMIKAZE
    CONFIG["ENABLE_EMP_WEAPONS"] = True
    CONFIG["ENABLE_LN2_WEAPON"] = True
    CONFIG["ENABLE_KAMIKAZE_WEAPON"] = True

    # Number of drones and battle duration
    NUM_BLUE = CONFIG["NUM_BLUE_DRONES"]
    NUM_RED = CONFIG["NUM_RED_DRONES"]
    MAX_TURNS = CONFIG["MAX_TURNS"]

    # Weather selection (CLEAR / MEDIUM / BAD)
    initial_weather = "CLEAR"
    #initial_weather = "BAD"

    # ----- Military campaign status -----
    campaign = CampaignState(
        turn=0,
        weather=initial_weather,
        logs=[],
        enemy_ai=None,
        drone_scenario=None,
        drone_mode_enabled=True,
        player_controls={
            #"player_personality": "defensive",   # "aggressive", "defensive", "deceptive"
            "player_personality": "aggressive",   # "aggressive", "defensive", "deceptive"
            "player_aggressiveness": 0.6,        # 0.0 – 1.0
            "player_pref_distance": 8.0,        # preferred distance
        },
    )

    # ----- Launch of the drone campaign -----
    print("\n=== RUN 1: Drone battle with EMP =", CONFIG['ENABLE_EMP_WEAPONS'], "===\n")
    run_drone_campaign(
        campaign_state=campaign,
        num_blue=NUM_BLUE,
        num_red=NUM_RED,
        max_turns=MAX_TURNS,
    )
    
    # 3D visualization if you keep your function
    visualize_drone_battle_3d(campaign)
    visualize_drone_2d_tracks(campaign)
    
    
    
    
    #...

    # ----- Example: second run without ... for comparison -----
    CONFIG["ENABLE_EMP_WEAPONS"] = False
    CONFIG["ENABLE_LN2_WEAPON"] = False
    CONFIG["ENABLE_KAMIKAZE_WEAPON"] = False
    
    campaign_no_emp = CampaignState(
        turn=0,
        weather=initial_weather,
        logs=[],
        enemy_ai=campaign.enemy_ai,  # we are using the trained enemy AI
        drone_scenario=None,
        drone_mode_enabled=True,
        player_controls=campaign.player_controls,
    )

    print("\n=== RUN 2: Drone battle with No EMP =", CONFIG['ENABLE_EMP_WEAPONS'], "===\n")
    run_drone_campaign(
        campaign_state=campaign_no_emp,
        num_blue=NUM_BLUE,
        num_red=NUM_RED,
        max_turns=MAX_TURNS,
    )

    visualize_drone_battle_3d(campaign)
    visualize_drone_2d_tracks(campaign)
    visualize_influence_heatmaps(campaign)

    #...

