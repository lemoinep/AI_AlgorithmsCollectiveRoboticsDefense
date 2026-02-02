# Drone Swarm Simulation


## **Introduction**

This repository extends prior developments to produce high-fidelity simulations of AI systems. Core objectives encompass visualization of emergent collective AI behaviors, implementation and validation of AI algorithms, and establishment of a dedicated research and education platform. Development remains in initial phases, with modular components to be incrementally constructed and integrated.


<p align="center">
<img src="Images/P0002.jpg" width="100%" />
</p>
<p align="center">
<img src="Images/P0003.jpg" width="100%" />
</p>


## **Detailed Python Program Descriptions**

### What the program actually does ?

- It simulates an environment where agents (drones) explore, identify, pursue, and "attack" targets, illustrating group strategies and dynamic task assignment in artificial intelligence.

- It allows visualization of collective agent behavior and dynamic resource (target) management in a 3D space.

- It provides a foundation for experimenting with AI algorithms for coordination, trajectory optimization, or target management in robotics or military scenarios.

### Potential Uses

This type of simulation can be used to:

- Test and develop multi-agent cooperation strategies.

- Visualize AI algorithms for collective robotics or defense.

- Serve as a basis for research or teaching projects on group dynamics, target management, or 3D trajectory planning.

- "Exploring and integrating targeted combat strategies using artificial intelligence. This project is currently in the development phase. I will gradually incorporate effective solutions as progress is made." 


### **Descriptions**

#### SimuDataLemAI.py: 

This program is a set of interactive 3D simulations written in Python using Pyglet and PyOpenGL to study collective robotics and defense-oriented AI strategies. The simulations model fleets of “drone avatars” (DA3DRs) evolving in a closed 3D space with both fixed and mobile targets, emphasizing group dynamics, target lifecycle management, and adaptive agent behaviour. They provide a playground for experimenting with multi-agent coordination, trajectory planning, and target allocation algorithms, and can be used for research, teaching, or visual demonstrations of AI in collective robotics and defense scenarios

#### SimuDronesCampaignLemAI_Level1.py 

This program is a configurable 3D drone battle simulator where blue and red squads fight on a 100×100 map with realistic movement, firing, battery usage, ammo limits, collision avoidance, and interception events. It includes adaptive enemy AI (aggressive, defensive, or deceptive) that learns from previous outcomes, Sun Tzu–inspired positional and influence models, and a tunable player controller for tactics like preferred distance and aggressiveness. Battles are turn-based (up to 5000 turns) with 3D Matplotlib animation for trajectories and loss counters, and the simulator also provides 2D ground tracks, altitude‑vs‑time plots, and influence heatmaps at key turns. Every shot and drone state is logged to export quantitative statistics (hit rates, distance distributions, battery usage, interception times), making the script suitable for experimenting with swarm tactics and drone warfare strategies simply by adjusting configuration parameters and drone counts in the CONFIG block and main entry point.

---

##### Conceptual Diagram: Program Structure and Enemy AI Flow

```
+----------------------------------------------------------------+
| SimuDronesCampaignLemAI_Level1.py                              |
| Configurable 3D drone battle + AI + visualization              |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| CONFIG                                                         |
| - Scenario (map, turns, N drones)                              |
| - Drone dynamics (speed, accel)                                |
| - Volume constraints (altitude, NFZ)                           |
| - Weather presets (wind, turbulence)                           |
| - Visualization / heatmap turns                                |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| Data & State Classes                                           |
| - DroneUnit                                                    |
|   (3D pose, HP, battery, ammo, dynamics state, team)           |
| - DroneScenarioState                                           |
|   (drones, turn, history,influence grids, shot_events)         |
| - CampaignState                                                |
|   (global logs, weather label, enemy_ai, player_controls)      |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| AI Components                                                  |
| - EnhancedEnemyAI                                              |
|   (personality, memory, recommend_drone_action)                |
| - PlayerDroneController                                        |
|   (uses player_controls:                                       |
|    personality, aggressiveness, preferred distance)            |
| - ChessSunTzuAI                                                |
|   (positional score per team)                                  |
| - GoSunTzuAI                                                   |
|   (influence grids, encirclement detection)                    |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| Environment & Physics                                          |
| - VolumeConstraints                                            |
|   (altitude limits, NFZ)                                       |
| - Weather                                                      |
|   (wind, turbulence, battery_factor)                           |
| - update_drone_dynamics                                        |
|   (inertia, accel, yaw, climb)                                 |
| - apply_weather,                                               |
|   apply_volume_constraints,                                    |
|   update_battery                                               |
| - apply_collision_avoidance                                    |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| Main Simulation Loop                                           |
| run_drone_campaign()                                           |
|  1. Check is_battle_over                                       |
|  2. Evaluate position                                          |
|     (ChessSunTzuAI, GoSunTzuAI)                                |
|  3. For each drone:                                            |
|     - Enemy: EnhancedEnemyAI                                   |
|     - Player: PlayerController                                 |
|  4. Apply collision avoidance                                  |
|  5. Integrate dynamics + env                                   |
|  6. Firing phase: hit/miss,                                    |
|     damage, interception,                                      |
|     log shot_events                                            |
|  7. Snapshot state to history                                  |
+-------------------------------+--------------------------------+
                                |
                                v
+-------------------------------+--------------------------------+
| Statistics & Visualization                                     |
| - export_battle_stats()                                        |
|   (hit rates, distance, battery, intercept times)              |
| - visualize_drone_battle_3d()                                  |
| - visualize_drone_2d_tracks()                                  |
| - visualize_influence_heatmaps()                               |
+----------------------------------------------------------------+

```

---

##### Enemy AI Decision Flow

```
+--------------------------------------------------------+
| EnhancedEnemyAI                                        |
| Role: tactical controller for red drones               |
+------------------------+-------------------------------+
                         |
                         v
+--------------------------------------------------------+
| 1. Update personality                                  |
|    after each battle                                   |
| observe_outcome()                                      |
|  - Append result to                                    |
|    memory (limited size)                               |
| decide_personality()                                   |
|  - Compute win rate                                    |
|  - If win_rate > 0.7:                                  |
|       personality = AGGRESSIVE                         |
|  - If win_rate < 0.3:                                  |
|       personality = DEFENSIVE                          |
|  - Else:                                               |
|       personality = DECEPTIVE                          |
+------------------------+-------------------------------+
                         |
                         v
+------------------------+-------------------------------+
| 2. Per-turn decision                                   |
|    for each enemy drone recommend_drone_action()       |
+------------------------+-------------------------------+
                         |
                         v
+------------------------+-------------------------------+
| 2.1 Perception                                         |
| - Collect enemy drones                                 |
|   (blue team) from DroneScenarioState                  |
| - If none: return zero velocity, no shot               |
+------------------------+-------------------------------+
                         |
                         v
+------------------------+-------------------------------+
| 2.2 Target selection                                   |
| - Choose closest blue  drone (min distance)            |
| - Compute distance                                     |
+------------------------+-------------------------------+
                         |
                         v
+--------------------------------------------------------+
| 2.3 Behavior by personality                            |
+------------------------+-------------------------------+
| aggressive             | defensive                     |
| - Compute vector       | - Compute desired stand-off   |
|   _vector_towards()    |   distance (weapon_range*0.7) |
|   (full speed)         | - If too close:               |
| - If in weapon range   |     _vector_away()            |
|   and ammo>0:          |   else:                       |
|   shoot = target       |     _vector_towards() (slower)|
+------------------------+-------------------------------+
| deceptive                                              |
| - If drone HP < 40:                                    |
|     _vector_away() (escape)                            |
| - Else:                                                |
|     _flank_vector() (lateral maneuver)                 |
| - Shooting thresholds slightly more conservative       |
+--------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------+
| 2.4 Output                                             |
| - desired velocity (vx,vy, vz) scaled by               |
|   dyn_max_speed / climb                                |
| - optional target to fire at if:                       |
|   distance within personality-specific firing window   |
|   AND ammo > 0                                         |
+--------------------------------------------------------+

```



Note: I will be adding the improvements soon...
In the meantime, I invite you to check out my other repository: https://github.com/lemoinep/SunTzuMilitaryCampaignSimulator

---

#### SimuDronesCampaignLemAI_Level2.py 

In this version, the simulation introduces **electromagnetic pulse (EMP) weapons** (systems similar to *SPEAR/THOR* ) as a new capability for certain drones. These high-power electromagnetic pulses do not destroy targets through kinetic impact, but instead temporarily disable their onboard electronic systems, reducing their health points, draining their batteries, and preventing them from moving or firing for several turns.

EMP weapons are modeled as area-of-effect systems: when activated, they affect all enemy drones within a configurable radius, making them particularly effective against clustered formations and drone swarms. An AI controller determines the optimal time to trigger these pulses by evaluating the local density of enemy drones, the distance to threats, and the health of the firing drone, ensuring that EMP shots are used sparingly and only when they can neutralize multiple opponents or turn the tide of a battle.

Tactically, this creates a new strategic dimension in the simulation: drones equipped with EMP weapons can sacrifice some kinetic firing opportunities to influence the battlefield, immobilizing enemy units, forcing them to reposition, and creating openings for conventional attacks. By enabling the ENABLE_EMP_WEAPONS option, users can directly compare battles with and without electromagnetic weapons and study how access to these non-kinetic area control tools alters strategies and outcomes at the drone swarm level.

---

#### SimuDronesCampaignLemAI_Level3.py 

Improving the previous simulation to effectively utilize the electromagnetic pulse weapon with drones. 

The idea is that since the **EMP effect decreases proportionally to \(1/r\)**, the drone formation should be optimized to **maximize its impact on enemy targets while keeping friendly drones safe**. Because the electromagnetic pulse radiates almost evenly in all directions, forming a **full circle around the target** is risky — some friendly drones would be exposed to fields strong enough to disrupt them due to positioning errors or secondary lobes.  

An **arc formation**, placed on the side **opposite the emitter** or slightly beyond the **danger zone**, allows multiple drones to **direct fire effectively** at the disabled enemies without entering the EMP’s effective radius. The tactic involves defining two distances:  

- An effective radius R_eff, where the EMP can neutralize enemy drones,  
- And a safety radius R_safe > R_eff, beyond which friendly drones remain unaffected.  

A specialized drone such as a **THOR** or **SPEAR** unit moves close enough so that the enemy group lies within R_eff, while the rest stay along the **arc** at or beyond R_safe.  

The EMP is triggered only when **enough enemies fall within the effective zone** and **no friendly drones are inside the safety radius**.  

For the next step, I will add a new weapon: a drone-mounted system that sprays liquid nitrogen onto enemy drones to induce technical failures. Because, liquid nitrogen itself is non-conductive, so it's not an "electrical" weapon but rather a mechanical/thermal one: rapid cooling can make plastics brittle, crack fragile parts, or cause certain components to malfunction, especially if sensitive areas are targeted (propellers, joints, exposed sensors, batteries). Therefore, it should be modeled as a short-range, cone-shaped weapon...



---

##### Conceptual Diagram: Program Structure and Enemy AI Flow

```
+---------------------------------------------------------------------------+
| SimuDronesCampaignLemAI_Level3.py                                         |
| 3D drone battle + EMP weapons + arc formation tactics + visualization     |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| CONFIG                                                                    |
| - Map, max turns, number of drones                                        |
| - Drone dynamics (speed, accel, climb, yaw)                               |
| - Volume constraints (altitude, NFZ)                                      |
| - Weather presets (wind, turbulence, battery factors)                     |
| - Weapon settings (kinetic, EMP, ranges, cooldowns)                       |
| - Formation parameters (R_eff, R_safe, arc geometry)                      |
| - Visualization / logging options                                         |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| Data & State Classes                                                      |
| - DroneUnit                                                               |
|   (3D pose, HP, battery, ammo, EMP status, team, role)                    |
| - DroneScenarioState                                                      |
|   (all drones, turn index, history, influence grids,                      |
|    shot_events, emp_events)                                               |
| - CampaignState                                                           |
|   (global logs, weather label, enemy_ai, player_controls,                 |
|    formation_planner, formation_targets)                                  |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| AI Components                                                             |
| - EnhancedEnemyAI                                                         |
|   (personality, memory, EMP-aware recommend_drone_action)                 |
| - PlayerDroneController                                                   |
|   (aggressiveness, preferred distance, EMP usage policy)                  |
| - FormationPlannerEMP                                                     |
|   - Detect clusters of enemy drones                                       |
|   - Select EMP carrier (THOR/SPEAR-like drone)                            |
|   - Compute arc positions at R_safe                                       |
|   - Assign roles: EMP_EMITTER vs ARC_SHOOTER                              |
| - ChessSunTzuAI                                                           |
|   (positional evaluation per team)                                        |
| - GoSunTzuAI                                                              |
|   (influence grids, encirclement / density analysis)                      |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| Environment, Physics & Weapons                                            |
| - VolumeConstraints (altitude limits, NFZ)                                |
| - Weather model (wind, turbulence, battery_factor)                        |
| - update_drone_dynamics (inertia, accel, yaw, climb)                      |
| - apply_weather, apply_volume_constraints, update_battery                 |
| - apply_collision_avoidance                                               |
| - Weapon handling:                                                        |
|   - Kinetic fire (hit/miss, damage, interception)                         |
|   - EMP propagation (1/r attenuation, R_eff, R_safe checks)               |
|   - Apply EMP effects (HP loss, battery drain, temporary disable)         |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| Main Simulation Loop: run_drone_campaign_level3()                         |
| 1. Check termination: is_battle_over                                      |
| 2. Evaluate global position (ChessSunTzuAI, GoSunTzuAI)                   |
| 3. Formation planning (FormationPlannerEMP)                               |
|    - Identify enemy clusters                                              |
|    - Update roles and desired arc around targets                          |
| 4. For each drone:                                                        |
|    - If enemy: EnhancedEnemyAI                                            |
|      (uses EMP awareness + personality)                                   |
|    - If player: PlayerDroneController                                     |
|      (tactics + EMP policy + formation hints)                             |
| 5. Apply collision avoidance                                              |
| 6. Integrate dynamics + environment                                       |
| 7. Weapons phase:                                                         |
|    - Resolve kinetic fire                                                 |
|    - Evaluate EMP trigger conditions                                      |
|      (enemies within R_eff, no friend in R_safe)                          |
|    - Apply EMP effects, log emp_events                                    |
| 8. Snapshot state to history                                              |
+-------------------------------+-------------------------------------------+
                                |
                                v
+-------------------------------+-------------------------------------------+
| Statistics & Visualization                                                |
| - export_battle_stats_level3()                                            |
|   (hit rates, distances, battery use, interception times, EMP usage)      |
| - visualize_drone_battle_3d_level3()                                      |
| - visualize_drone_2d_tracks_level3()                                      |
| - visualize_influence_heatmaps_level3()                                   |
| - Optional overlays for R_eff, R_safe and arc positions                   |
+--------------------------------------------------------------------------+

```

---

##### Enemy AI Decision Flow

```
+-------------------------------------------------------------------+
| EnhancedEnemyAI (Level 3)                                         |
| Role: tactical controller for red drones with EMP/formation logic |
+---------------------------+---------------------------------------+
                            |
                            v
+-------------------------------------------------------------------+
| 1. Update personality after each battle                           |
|   observe_outcome()                                               |
|   - Append battle result to memory (limited size)                 |
|   decide_personality()                                            |
|   - Compute win rate                                              |
|   - If win_rate > 0.7  -> AGGRESSIVE                              |
|   - If win_rate < 0.3  -> DEFENSIVE                               |
|   - Else               -> DECEPTIVE                               |
+---------------------------+---------------------------------------+
                            |
                            v
+---------------------------+---------------------------------------+
| 2. Per-turn decision for each enemy drone                         |
|   recommend_drone_action_level3()                                 |
+---------------------------+---------------------------------------+
                            |
                            v
+---------------------------+---------------------------------------+
| 2.1 Perception                                                    |
|   - Read DroneScenarioState                                       |
|   - Collect visible blue drones                                   |
|   - Check local cluster density around candidate EMP targets      |
|   - If no blue drones:                                            |
|       return zero velocity, no shot, no EMP                       |
+---------------------------+---------------------------------------+
                            |
                            v
+---------------------------+--------------------------------------+
| 2.2 Target and role selection                                    |
|   - Choose main kinetic target (closest or tactically relevant)  |
|   - If drone is EMP carrier:                                     |
|       * Evaluate cluster size within R_eff                       |
|       * Check distance to cluster center                         |
|       * Mark candidate_EMP_trigger (bool)                        |
|   - Else (non-EMP drone):                                        |
|       * Use formation hints (arc position at R_safe)             |
+---------------------------+--------------------------------------+
                            |
                            v
+--------------------------------------------------------------------+
| 2.3 Behavior by personality and role                               |
+---------------------------+-----------------+----------------------+
| aggressive                | defensive       | deceptive            |
| - EMP carrier:            | - EMP carrier:  | - EMP carrier:       |
|   * Rush toward cluster   |   * Stay just   |   * Fake approaches  |
|     center to reach       |     outside     |     and lateral      |
|     R_eff quickly         |     R_safe      |     moves before     |
|   * If cluster >= N_min   |   * Trigger EMP |     entering R_eff   |
|     and no friend in      |     only when   |   * Trigger EMP      |
|     R_safe: trigger EMP   |     clearly safe|     opportunistically|
| - Non-EMP:                | - Non-EMP:      | - Non-EMP:           |
|   * Full-speed vector     |   * Keep stand- |   * Flanking arcs    |
|     _vector_towards()     |     off at      |     and irregular    |
|   * If in weapon range    |     R_safe      |     spacing on arc   |
|     and ammo>0: fire      |   * Prioritize  |   * More             |
|                           |     survival    |     conservative     |
|                           |                 |     fire thresholds  |
+---------------------------+-----------------+----------------------+
                            |
                            v
+---------------------------+--------------------------------------+
| 2.4 EMP trigger logic (EMP carrier only)                         |
|   - If candidate_EMP_trigger is true                             |
|   - And enemies_in_R_eff >= N_min                                |
|   - And no friendly drone inside R_safe                          |
|   - And EMP not on cooldown:                                     |
|       * fire_emp_pulse()                                         |
|       * Mark affected drones as EMP-disabled for k turns         |
+---------------------------+--------------------------------------+
                            |
                            v
+---------------------------+--------------------------------------+
| 2.5 Output                                                       |
|   - desired velocity (vx, vy, vz) within dynamics limits         |
|   - optional kinetic target to fire at                           |
|   - optional EMP trigger flag (for EMP carrier)                  |
+------------------------------------------------------------------+

```

---


## For more information

<img src="Images/Z20260122_000001.jpg" width="100%" />
<img src="Images/Z20260122_000002.jpg" width="100%" />
<img src="Images/Z20260122_000003.jpg" width="100%" />
<img src="Images/Z20260122_000004.jpg" width="100%" />
<img src="Images/Z20260122_000005.jpg" width="100%" />
<img src="Images/Z20260122_000006.jpg" width="100%" />
<img src="Images/Z20260122_000007.jpg" width="100%" />
<img src="Images/Z20260122_000008.jpg" width="100%" />
<img src="Images/Z20260122_000009.jpg" width="100%" />
<img src="Images/Z20260122_000010.jpg" width="100%" />
<img src="Images/Z20260122_000011.jpg" width="100%" />
<img src="Images/Z20260122_000012.jpg" width="100%" />

---

<img src="Images/Z20260123_000001.jpg" width="100%" />
<img src="Images/Z20260123_000002.jpg" width="100%" />
<img src="Images/Z20260123_000003.jpg" width="100%" />
<img src="Images/Z20260123_000004.jpg" width="100%" />
<img src="Images/Z20260123_000005.jpg" width="100%" />
<img src="Images/Z20260123_000006.jpg" width="100%" />
<img src="Images/Z20260123_000007.jpg" width="100%" />
<img src="Images/Z20260123_000008.jpg" width="100%" />
<img src="Images/Z20260123_000009.jpg" width="100%" />
<img src="Images/Z20260123_000010.jpg" width="100%" />
<img src="Images/Z20260123_000011.jpg" width="100%" />
<img src="Images/Z20260123_000012.jpg" width="100%" />
<img src="Images/Z20260123_000013.jpg" width="100%" />

