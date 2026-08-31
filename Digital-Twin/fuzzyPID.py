import math
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
Kp0 = 100.0
Ki0 = 550.0
Kd0 = 1.5
uI_last = 0.0
error_last = 0.0
dy_dt = 0.0

Kp_gain = 70.0
Ki_gain = 350.0
Kd_gain = 0.5

### FILTER PARAMETERS ###
alpha_filter = 0.5
y_filtered_last = 0.0

y_target = 50.0

### FUZZIFICATION ###
def triangle_fuzz(x, a, b, c):
    """
    x - input
    a - triangle start
    b - triangle peak
    c - triangle end
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0

def fuzzify_input(x_norm):
    x_norm = max(-1.0, min(1.0, x_norm))

    return {
        'NB': triangle_fuzz(x_norm, -1.0, -1.0, -0.5),
        'NS': triangle_fuzz(x_norm, -1.0, -0.5, 0.0),
        'Z': triangle_fuzz(x_norm, -0.5, 0.0, 0.5),
        'PS': triangle_fuzz(x_norm, 0.0, 0.5, 1.0),
        'PB': triangle_fuzz(x_norm, 0.5, 1.0, 1.0)
    }

### FUZZY RULES ###
def get_fuzzy_attr(error, error_last):
    error_norm = error / 100.0
    der_error_norm = (error - error_last) / dt / 1500.0

    error_linguistic = fuzzify_input(error_norm)
    der_error_linguistic = fuzzify_input(der_error_norm)

    rules_Kp = {
        'NB': {'NB': 'PB', 'NS': 'PB', 'Z': 'PM', 'PS': 'PM', 'PB': 'PS'},
        'NS': {'NB': 'PB', 'NS': 'PM', 'Z': 'PM', 'PS': 'PS', 'PB': 'PS'},
        'Z': {'NB': 'PM', 'NS': 'PS', 'Z': 'Z', 'PS': 'NS', 'PB': 'NM'},
        'PS': {'NB': 'NS', 'NS': 'NS', 'Z': 'NM', 'PS': 'NM', 'PB': 'NB'},
        'PB': {'NB': 'NS', 'NS': 'NM', 'Z': 'NM', 'PS': 'NB', 'PB': 'NB'},
    }

    rules_Ki = {
        'NB': {'NB': 'NB', 'NS': 'NB', 'Z': 'NM', 'PS': 'NM', 'PB': 'Z'},
        'NS': {'NB': 'NB', 'NS': 'NM', 'Z': 'NS', 'PS': 'NS', 'PB': 'Z'},
        'Z': {'NB': 'NM', 'NS': 'NS', 'Z': 'Z', 'PS': 'PS', 'PB': 'PM'},
        'PS': {'NB': 'Z', 'NS': 'PS', 'Z': 'PS', 'PS': 'PM', 'PB': 'PB'},
        'PB': {'NB': 'Z', 'NS': 'PM', 'Z': 'PM', 'PS': 'PB', 'PB': 'PB'},
    }

    rules_Kd = {
        'NB': {'NB': 'PS', 'NS': 'NS', 'Z': 'NB', 'PS': 'NB', 'PB': 'NB'},
        'NS': {'NB': 'PS', 'NS': 'NS', 'Z': 'NB', 'PS': 'NM', 'PB': 'NM'},
        'Z': {'NB': 'Z', 'NS': 'NS', 'Z': 'NS', 'PS': 'NS', 'PB': 'Z'},
        'PS': {'NB': 'NM', 'NS': 'NM', 'Z': 'NB', 'PS': 'NS', 'PB': 'PS'},
        'PB': {'NB': 'NB', 'NS': 'NB', 'Z': 'NB', 'PS': 'NS', 'PB': 'PS'},
    }

    centers = {'NB': -1.0, 'NM': -0.66, 'NS': -0.33, 'Z': 0.0, 'PS': 0.33, 'PM': 0.66, 'PB': 1.0}

    num_p, den_p = 0.0, 0.0
    num_i, den_i = 0.0, 0.0
    num_d, den_d = 0.0, 0.0

    for e_key, e_val in error_linguistic.items():
        for de_key, de_val in der_error_linguistic.items():
            # Siła aktywacji reguły (T-norma minimum)
            activation = min(e_val, de_val)

            if activation > 0:
                # Wyciągnięcie reguł dla danej kombinacji
                out_p = rules_Kp[e_key][de_key]
                out_i = rules_Ki[e_key][de_key]
                out_d = rules_Kd[e_key][de_key]

                # Akumulacja do licznika i mianownika (wyostrzanie)
                num_p += activation * centers[out_p]
                den_p += activation

                num_i += activation * centers[out_i]
                den_i += activation

                num_d += activation * centers[out_d]
                den_d += activation

    # Zabezpieczenie przed dzieleniem przez zero i obliczenie wyjścia
    delta_Kp_norm = (num_p / den_p) if den_p != 0 else 0.0
    delta_Ki_norm = (num_i / den_i) if den_i != 0 else 0.0
    delta_Kd_norm = (num_d / den_d) if den_d != 0 else 0.0

    return delta_Kp_norm, delta_Ki_norm, delta_Kd_norm

### SIMULATION ###
for step in range(steps):
    if step == 1000: y_target = 70.0
    if step == 2000: y_target = 0.0
    # y_target = math.sin(step * math.pi/1500) * 30 + 50

    y_measured = y + np.random.uniform(-0.5, 0.5)

    y_filtered = alpha_filter * y_measured + (1 - alpha_filter) * y_filtered_last
    y_filtered_last = y_filtered

    error = y_target - y_filtered
    e_array[step] = error

    Kp_delta, Ki_delta, Kd_delta = get_fuzzy_attr(error, error_last)

    Kp = max(0, Kp0 + Kp_delta * Kp_gain)
    Ki = max(0, Ki0 + Ki_delta * Ki_gain)
    Kd = max(0, Kd0 + Kd_delta * Kd_gain)

    uP = Kp * error
    uI = uI_last + Ki * dt * (error_last + error) / 2
    if uI > 3999: uI = 3999
    if uI < -3999: uI = -3999
    uD = Kd * (error - error_last) / dt

    u = uP + uI + uD

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