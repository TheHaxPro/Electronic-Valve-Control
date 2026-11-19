import numpy as np
import matplotlib.pyplot as plt

u = np.array([])
y = np.array([])
y_target = np.array([])
time_array = np.array([])

# data = np.loadtxt('data_realtime.csv', delimiter=',')
data = np.loadtxt('Identification/1char_stat.txt', delimiter=',')

u = data[:, 1]
y = data[:, 2]
y_target = data[:, 3]
time_array = data[:, 0]

time_array = np.arange(len(u)) / 1000.0

print(y_target)

### PLOT ###
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

# Górny wykres - y(t)
ax1.plot(time_array, y_target, 'r--', label='Target [%]')
ax1.plot(time_array, y, 'b-', label='Real position [%]')
ax1.set_title('Weryfikacja algorytmu SMC na obiekcie')
ax1.set_ylabel('Pozycja [%]')
ax1.set_xlabel('Czas [s]')
ax1.legend()
ax1.grid(True)

# Dolny wykres - u(t)
ax2.plot(time_array, u, 'g-', label='PWM signal (U)')
ax2.set_xlabel('Czas [s]')
ax2.set_ylabel('Sterowanie PWM')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()



# import serial
# import time
# import csv
#
# # UART
# PORT = 'COM4'
# BAUDRATE = 115200
#
# ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
#
# start_time = time.monotonic()
#
# # Plik CSV
# csv_file = open('data_realtime.csv', 'w', newline='')
# writer = csv.writer(csv_file)
# writer.writerow(['time', 'u', 'TPS1', 'target'])
#
# def serial_reader():
#     while True:
#         try:
#             line = ser.readline().decode('ascii', errors='ignore').strip()
#
#             if not line:
#                 continue
#
#             parts = line.split(',')
#
#             if len(parts) != 3:
#                 continue
#
#             u = float(parts[0])
#             tps = float(parts[1])
#             target = float(parts[2])
#
#             t = time.monotonic() - start_time
#
#             with lock:
#                 timestamps.append(t)
#                 u_values.append(u)
#                 tps_values.append(tps)
#                 target_values.append(target)
#
#             writer.writerow([t, u, tps, target])
#
#         except (ValueError, serial.SerialException):
#             continue
#
#         except KeyboardInterrupt:
#             print("\n[*] Zamykanie programu...")
#             break
#
# serial_reader()