# Author(s): Dr. Patrick Lemoine

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. Core structures: drones + campaign
# ============================================================

class DroneStatus(Enum):
    ACTIVE = "ACTIVE"
    DESTROYED = "DESTROYED"          # kept for completeness
    OUT_OF_BATTERY = "OUT_OF_BATTERY"
    INTERCEPTED = "INTERCEPTED"      # shot down and crashed to the ground


@dataclass
class DroneUnit:
    """
    Basic drone representation for the drone battle scenario.
    Teams use "player" / "enemy".
    """
    id: str
    team: str               # "player" or "enemy"
    x: float
    y: float
    z: float = 0.0          # altitude
    hp: float = 100.0
    battery: float = 1000.0  # electric battery charge
    ammo: int = 4
    sensor_range: float = 30.0
    weapon_range: float = 20.0
    max_speed: float = 5.0
    status: DroneStatus = DroneStatus.ACTIVE

    def is_alive(self) -> bool:
        return self.status == DroneStatus.ACTIVE and self.hp > 0

    def distance_to(self, other: "DroneUnit") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class DroneScenarioState:
    """
    State holder for the drone scenario.
    """
    drones: List[DroneUnit] = field(default_factory=list)
    turn_index: int = 0
    max_turns: int = 500
    weather: str = "CLEAR"  # CLEAR / MEDIUM / BAD
    map_width: int = 100
    map_height: int = 100
    influence_grid_player: List[List[float]] = field(default_factory=list)
    influence_grid_enemy: List[List[float]] = field(default_factory=list)

    # history[turn][drone_id] = (x, y, z, hp, battery, status)
    history: List[Dict[str, Tuple[float, float, float, float, float, DroneStatus]]] = field(default_factory=list)

    def init_influence_grids(self, grid_size: int = 20) -> None:
        self.influence_grid_player = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.influence_grid_enemy = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]

    def get_team_drones(self, team: str) -> List[DroneUnit]:
        return [d for d in self.drones if d.team == team and d.is_alive()]

    def is_battle_over(self) -> Optional[str]:
        """
        Returns:
            "player"  : player wins
            "enemy"   : enemy wins
            "draw"    : both destroyed
            "timeout" : max turns reached
            None      : battle continues
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
    """
    Minimal campaign state compatible with the drone scenario.
    """
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
# 2. Drone interception helper
# ============================================================

def drone_intercepted(drone: DroneUnit, campaign_state: Optional[CampaignState] = None) -> None:
    """
    Handle interception: drone is shot down, falls to the ground and becomes 'INTERCEPTED'.
    """
    drone.hp = 0.0
    drone.status = DroneStatus.INTERCEPTED
    drone.z = 0.0  # on the ground
    if campaign_state is not None:
        campaign_state.log(f"[Drone] {drone.id} intercepted and crashed to the ground")


# ============================================================
# 3. EnhancedEnemyAI (simplified core + drone extension)
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
        self,
        drone: DroneUnit,
        state: DroneScenarioState
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
            factor * dx / norm * drone.max_speed,
            factor * dy / norm * drone.max_speed,
            factor * dz / norm * drone.max_speed,
        )

    def _vector_away(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = drone.x - target.x, drone.y - target.y, drone.z - target.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.max_speed,
            factor * dy / norm * drone.max_speed,
            factor * dz / norm * drone.max_speed,
        )

    def _flank_vector(self, drone: DroneUnit, target: DroneUnit):
        dx, dy = target.x - drone.x, target.y - drone.y
        norm = math.hypot(dx, dy) + 1e-6
        fx, fy = -dy / norm, dx / norm
        return fx * drone.max_speed, fy * drone.max_speed, 0.0


# ============================================================
# 4. ChessSunTzuAI + GoSunTzuAI
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
# 5. PlayerDroneController
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
        self,
        drone: DroneUnit,
        state: DroneScenarioState
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
            factor * dx / norm * drone.max_speed,
            factor * dy / norm * drone.max_speed,
            factor * dz / norm * drone.max_speed,
        )

    def _vector_away(self, drone: DroneUnit, target: DroneUnit, factor: float = 1.0):
        dx, dy, dz = drone.x - target.x, drone.y - target.y, drone.z - target.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        return (
            factor * dx / norm * drone.max_speed,
            factor * dy / norm * drone.max_speed,
            factor * dz / norm * drone.max_speed,
        )

    def _flank_vector(self, drone: DroneUnit, target: DroneUnit):
        dx, dy = target.x - drone.x, target.y - drone.y
        norm = math.hypot(dx, dy) + 1e-6
        fx, fy = -dy / norm, dx / norm
        return fx * drone.max_speed, fy * drone.max_speed, 0.0


# ============================================================
# 6. Drone initialization helper (variable counts)
# ============================================================

def create_drones(num_blue: int, num_red: int) -> List[DroneUnit]:
    """
    Creates num_blue player drones (left side) and num_red enemy drones (right side),
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
            )
        )

    return drones


# ============================================================
# 7. Collision avoidance helper
# ============================================================

def apply_collision_avoidance(
    ds: DroneScenarioState,
    min_dist: float = 3.0,
    strength: float = 1.5
) -> Dict[str, Tuple[float, float, float]]:
    """
    Simple collision avoidance:
    - For each pair of alive drones closer than min_dist,
      apply a repulsive vector.
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
                # Repulsive force
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
# 8. Main loop: run_drone_campaign (parameterized)
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
        map_width=100,
        map_height=100,
    )
    ds.drones = create_drones(num_blue=num_blue, num_red=num_red)
    ds.init_influence_grids()

    chess_ai = ChessSunTzuAI(map_width=ds.map_width, map_height=ds.map_height)
    go_ai = GoSunTzuAI(grid_size=20, map_width=ds.map_width, map_height=ds.map_height)

    enemy_ai: EnhancedEnemyAI = campaign_state.enemy_ai
    player_ai = PlayerDroneController(campaign_state.player_controls)

    campaign_state.log(
        f"=== Drone Campaign: {num_red} enemy drones vs {num_blue} player drones ==="
    )

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
                (vx, vy, vz), target = enemy_ai.recommend_drone_action(d, ds)
            else:
                (vx, vy, vz), target = player_ai.recommend_action(d, ds)

            planned_moves[d.id] = (vx, vy, vz)
            if target is not None:
                planned_shots.append((d, target))

        # Collision avoidance
        extra_vel = apply_collision_avoidance(ds, min_dist=3.0, strength=1.5)

        # Movement phase
        for d in ds.drones:
            if not d.is_alive():
                continue
            vx, vy, vz = planned_moves.get(d.id, (0.0, 0.0, 0.0))

            if d.id in extra_vel:
                evx, evy, evz = extra_vel[d.id]
                vx += evx
                vy += evy
                vz += evz

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if speed > d.max_speed:
                factor = d.max_speed / (speed + 1e-6)
                vx *= factor
                vy *= factor
                vz *= factor

            d.x += vx
            d.y += vy
            d.z += vz

            # Clamp Z for alive drones
            if d.is_alive() and d.z < 0.0:
                d.z = 0.0

            d.battery -= speed * 0.3
            if d.battery <= 0 and d.status == DroneStatus.ACTIVE:
                d.status = DroneStatus.OUT_OF_BATTERY
                campaign_state.log(f"[Drone] {d.id} out of battery")

        # Firing phase
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

            if random.random() < hit_prob:
                dmg = random.uniform(60, 120)
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

        snapshot: Dict[str, Tuple[float, float, float, float, float, DroneStatus]] = {}
        for d in ds.drones:
            snapshot[d.id] = (d.x, d.y, d.z, d.hp, d.battery, d.status)
        ds.history.append(snapshot)

    campaign_state.drone_scenario = ds


# ============================================================
# 9. 3D visualization (intercepted drones in black + counters)
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
    statuses: Dict[str, List[DroneStatus]] = {d: [] for d in drone_ids}
    teams: Dict[str, str] = {}

    for snap in ds.history:
        for d_id, (x, y, z, hp, battery, status) in snap.items():
            xs[d_id].append(x)
            ys[d_id].append(y)
            zs[d_id].append(z)
            statuses[d_id].append(status)

    for d in ds.drones:
        teams[d.id] = d.team

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    player_scatter = ax.scatter([], [], [], c="blue", label="Player (alive)")
    enemy_scatter = ax.scatter([], [], [], c="red", label="Enemy (alive)")
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
        0, 0, 30,
        "",
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    def update(frame: int):
        px, py, pz = [], [], []
        ex, ey, ez = [], [], []
        ix, iy, iz = [], [], []

        alive_count = 0
        red_intercepted = 0
        blue_intercepted = 0

        for d_id in drone_ids:
            if frame >= len(xs[d_id]):
                continue
            x = xs[d_id][frame]
            y = ys[d_id][frame]
            z = zs[d_id][frame]
            st = statuses[d_id][frame]
            team = teams[d_id]

            if st == DroneStatus.INTERCEPTED:
                ix.append(x); iy.append(y); iz.append(z)
                if team == "enemy":
                    red_intercepted += 1
                else:
                    blue_intercepted += 1
            else:
                if st == DroneStatus.ACTIVE:
                    alive_count += 1
                if team == "player":
                    px.append(x); py.append(y); pz.append(z)
                else:
                    ex.append(x); ey.append(y); ez.append(z)

        player_scatter._offsets3d = (px, py, pz)
        enemy_scatter._offsets3d = (ex, ey, ez)
        intercepted_scatter._offsets3d = (ix, iy, iz)

        info_text.set_text(
            f"Alive: {alive_count}\n"
            f"Red intercepted: {red_intercepted}\n"
            f"Blue intercepted: {blue_intercepted}"
        )

        return player_scatter, enemy_scatter, intercepted_scatter, info_text

    anim = FuncAnimation(fig, update, frames=turns, interval=200, blit=False)
    plt.show()


# ============================================================
# 10. Entry point (configurable numbers)
# ============================================================

if __name__ == "__main__":
    cs = CampaignState()
    cs.enemy_ai = EnhancedEnemyAI()
    #cs.enemy_ai.personality = "aggressive"  
    cs.player_controls = {
        "player_personality": "defensive",
        "player_aggressiveness": 0.95,
        "player_pref_distance": 10.0,
    }
    cs.drone_mode_enabled = True

    num_blue_drones =10
    num_red_drones = 10
    max_turns = 5000

    if cs.drone_mode_enabled:
        run_drone_campaign(
            cs,
            num_blue=num_blue_drones,
            num_red=num_red_drones,
            max_turns=max_turns,
        )
        visualize_drone_battle_3d(cs)
