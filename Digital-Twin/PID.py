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
e_array = np.zeros(steps)

ISE = 0
IAE = 0
ITAE = 0
ISC = 0

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
Td = 0.015
Kff = 1.0
uI_last = 0.0
error_last = 0.0
dy_dt = 0.0

### FILTER PARAMETERS ###
alpha_filter = 0.5
y_filtered_last = 0.0

y_target = 50.0

### SIMULATION ###
for step in range(steps):
    if step == 1000: y_target = 70.0
    if step == 2000: y_target = 0.0

    y_measured = y + np.random.uniform(-0.5, 0.5)

    y_filtered = alpha_filter * y_measured + (1 - alpha_filter) * y_filtered_last
    y_filtered_last = y_filtered

    error = y_target - y_filtered
    e_array[step] = error

    uP = K * error
    uI = uI_last + K / Ti * dt * (error_last + error) / 2
    if uI > 3999: uI = 3999
    if uI < -3999: uI = -3999
    uD = K * Td * (error - error_last) / dt

    ## Inverse Model Feedforward
    uFF = Kff * (spring_offset + y_target * K_spring)

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

    ITAE += step * dt * np.abs(error) * dt

ISE = np.sum(e_array ** 2) * dt
IAE = np.sum(np.abs(e_array)) * dt
ISC = np.sum(u_array ** 2) * dt


### PLOT ###
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

# Górny wykres - y(t)
ax1.plot(time_array, y_target_array, 'r--', label='Target [%]')
ax1.plot(time_array, y_array, 'b-', label='Real position [%]')
ax1.set_title('Symulacja algorytmu PID - Skoki wartości zadanej')
ax1.set_ylabel('Pozycja [%]')
ax1.set_xlabel('Czas [s]')
ax1.legend()
ax1.grid(True)

# Dolny wykres - u(t)
ax2.plot(time_array, u_array, 'g-', label='PWM signal (U)')
ax2.set_xlabel('Czas [s]')
ax2.set_ylabel('Sterowanie PWM')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print("ISE: ", round(ISE, 2))
print("IAE: ", round(IAE, 2))
print("ITAE: ", round(ITAE, 2))
print("ISC: ", round(ISC, 2))