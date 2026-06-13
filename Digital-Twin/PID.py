from math import copysign
import numpy as np
import matplotlib.pyplot as plt

### SIMULATION ###
dt = 0.001
sim_time = 3.0
steps = int(sim_time/dt)

time_array = np.arange(0,sim_time,dt)
u_array = np.zeros(steps)
y_target_array = np.zeros(steps)
y_array = np.zeros(steps)

### OBJECT PARAMETERS ###
spring_offset = 1480
K_spring = 4.8

static_friction_limit = 40.0
coulomb_friction = static_friction_limit / 2.0
B_friction = 2.0

J_obj = 0.05
y = 0.0

### PID PARAMETERS ###
K = 50.0
Ti = 5.0
Td = 0.02
uI_last = 0.0
error_last = 0.0
dy_dt = 0.0



y_target = 50.0

### SIMULATION ###
for step in range(steps):
    if step == 1000: y_target = 70.0
    if step == 2000: y_target = 0.0

    y = y + np.random.uniform(-0.5, 0.5)

    error = y_target - y

    uP = K * error
    uI = uI_last + K / Ti * dt * (error_last + error) / 2
    uD = K * Td * (error - error_last) / dt

    ## Inverse Model Feedforward
    uFF = spring_offset + y_target * K_spring

    u = uP + uI + uD + uFF

    if u > 3999: u = 3999
    if u < 0: u = 0

    motor_torque = u

    spring_torque = spring_offset + y * K_spring

    viscous_friction = B_friction * dy_dt
    if dy_dt > 0:
        friction_torque = viscous_friction + coulomb_friction
    elif dy_dt < 0:
        friction_torque = viscous_friction - coulomb_friction
    else:
        move_force = motor_torque - spring_torque
        if abs(move_force) > static_friction_limit:
            friction_torque = copysign(coulomb_friction, move_force)
        else:
            friction_torque = move_force

    # acceleration
    d2y_dt2 = (motor_torque - spring_torque - friction_torque) / J_obj

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

    error_last = error
    uI_last = uI

    u_array[step] = u
    y_target_array[step] = y_target
    y_array[step] = y


### PLOT ###
plt.figure(figsize=(10, 6))
plt.plot(time_array, y_target_array, 'r--', label='Target [%]')
plt.plot(time_array, y_array, 'b-', label='Real position [%]')
plt.title('Electronic Valve Control Simulation')
plt.xlabel('Time [s]')
plt.ylabel('Position [%]')
plt.legend()
plt.grid(True)
plt.show()