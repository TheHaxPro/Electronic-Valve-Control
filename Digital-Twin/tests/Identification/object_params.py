import math
from math import copysign
import numpy as np
import matplotlib.pyplot as plt

### SIMULATION ###
dt = 0.012
sim_time = 3.0
steps = int(sim_time/dt)

time_array = np.arange(0,sim_time,dt)
u_array = np.zeros(steps)
y_array = np.zeros(steps)

### OBJECT PARAMETERS ###
spring_offset = 900
K_spring = 5.2

static_friction_limit = 340.0
coulomb_friction = 200.0
# coulomb_friction = static_friction_limit / 2.0
B_friction = 3.8

J_obj = 0.5
velocity_epsilon = 0.1

y = 0.0
dy_dt = 0.0
u = 1520

data = np.loadtxt('3char_dyn.txt', delimiter=',')

y_real = data[:, 2]
if len(y_real) < steps:
    y_real = np.pad(
        y_real,
        (0, steps - len(y_real)),
        mode='edge'
    )

### SIMULATION ###
for step in range(steps):

    if u > 3999: u = 3999
    if u < 0: u = 0

    motor_torque = u

    spring_torque = spring_offset + y * K_spring

    viscous_friction = B_friction * dy_dt

    move_force = motor_torque - spring_torque

    if abs(dy_dt) <= velocity_epsilon:
        if abs(move_force) <= static_friction_limit:
            dy_dt = 0.0
            d2y_dt2 = 0.0 # acceleration
        else:
            friction_torque = copysign(coulomb_friction, move_force)
            d2y_dt2 = (move_force - friction_torque) / J_obj # acceleration
    else:
        friction_torque = viscous_friction + copysign(coulomb_friction, dy_dt)
        d2y_dt2 = (move_force - friction_torque) / J_obj # acceleration

    # speed
    dy_dt += d2y_dt2 * dt

    # position
    dy = dy_dt * dt

    y += dy

    if y > 100.0:
        y = 100.0
        dy_dt = 0.0
    if y < 0.0:
        y = 0.0
        dy_dt = 0.0

    u_array[step] = u
    y_array[step] = y

### PLOT ###
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
# Górny wykres - y(t)
ax1.plot(time_array, y_real, 'r--', label='Przepustnica')
ax1.plot(time_array, y_array, 'b-', label='Symulacja')
ax1.set_title('Porównanie parametrów symulacji z rzeczywistą przepustnicą')
ax1.set_ylabel('Pozycja [%]')
ax1.set_xlabel('Czas [s]')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()