# Drone Swarm Simulation


## **Introduction**

This repository extends prior developments to produce high-fidelity simulations of AI systems. Core objectives encompass visualization of emergent collective AI behaviors, implementation and validation of AI algorithms, and establishment of a dedicated research and education platform. Development remains in initial phases, with modular components to be incrementally constructed and integrated.


<p align="center">
<img src="Images/P0002.jpg" width="100%" />
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

- **SimuDataLemAI.py**: a set of interactive 3D simulations written in Python using Pyglet and PyOpenGL to study collective robotics and defense-oriented AI strategies. The simulations model fleets of “drone avatars” (DA3DRs) evolving in a closed 3D space with both fixed and mobile targets, emphasizing group dynamics, target lifecycle management, and adaptive agent behaviour. They provide a playground for experimenting with multi-agent coordination, trajectory planning, and target allocation algorithms, and can be used for research, teaching, or visual demonstrations of AI in collective robotics and defense scenarios

- **SimuDronesCampaignLemAI_Level1.py** : is a configurable 3D drone battle simulator where blue and red squads fight on a 100×100 map with realistic movement, firing, battery usage, ammo limits, collision avoidance, and interception events. It includes adaptive enemy AI (aggressive, defensive, or deceptive) that learns from previous outcomes, Sun Tzu–inspired positional and influence models, and a tunable player controller for tactics like preferred distance and aggressiveness. Battles are turn-based (up to 5000 turns) with 3D Matplotlib animation for trajectories and loss counters, making the script suitable for experimenting with swarm tactics and drone warfare strategies simply by adjusting drone counts and parameters in the main entry point.

Note: I will be adding the improvements soon...
In the meantime, I invite you to check out my other repository: https://github.com/lemoinep/SunTzuMilitaryCampaignSimulator
