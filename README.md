This program is an interactive 3D simulation written in Python, using the Pyglet and PyOpenGL libraries. It features a fleet of "drone avatars" (DA3DRs) operating within a closed space alongside both fixed and mobile targets. This platform is designed to explore and integrate targeted combat strategies using artificial intelligence, with a focus on group dynamics, target management, and adaptive agent behavior.

== Main Features

Real-time 3D simulation: Displays a 3D scene containing drones (DA3DRs), fixed targets, and mobile targets, with mouse-based navigation (rotation, zoom).

Autonomous agents (DA3DRs): 100 drones are initialized at random positions and move autonomously in space. Each drone selects a target to pursue based on proximity and the target’s status.

== Fixed and mobile targets:

Fixed targets: Remain stationary. When a drone stays close enough for a certain period, the target becomes inactive and then gradually disappears.

Mobile targets: Move in space in a random direction, bouncing off the boundaries. They also become inactive if a drone stays nearby long enough, then gradually disappear.

Target lifecycle management: Inactive targets shrink and are removed from the simulation after a delay. When all targets are inactive, new mobile targets are randomly generated to keep the simulation active.

Intelligent target assignment: Drones are evenly distributed across active mobile targets, avoiding redundancy and optimizing coverage.

== Visualization:

- Drones are drawn as red spheres (exploring) or yellow spheres (attacking).
- Drone trajectories are displayed.
- Fixed and mobile targets are represented by green spheres (active) or black spheres (inactive).
- 3D axes and grid for spatial reference.


== What the program actually does

- It simulates an environment where agents (drones) explore, identify, pursue, and "attack" targets, illustrating group strategies and dynamic task assignment in artificial intelligence.

- It allows visualization of collective agent behavior and dynamic resource (target) management in a 3D space.

- It provides a foundation for experimenting with AI algorithms for coordination, trajectory optimization, or target management in robotics or military scenarios.

== Potential Uses

This type of simulation can be used to:

- Test and develop multi-agent cooperation strategies.

- Visualize AI algorithms for collective robotics or defense.

- Serve as a basis for research or teaching projects on group dynamics, target management, or 3D trajectory planning.

- "Exploring and integrating targeted combat strategies using artificial intelligence. This project is currently in the development phase. I will gradually incorporate effective solutions as progress is made." 

