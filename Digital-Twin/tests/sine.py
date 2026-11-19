import math

# Parametry Twojego sinusa
omega = 2.0 * math.pi / 3.0
amplitude = 30.0
offset = 50.0

# Parametry czasowe dla STM32
sample_rate_hz = 1000  # Częstotliwość pętli sterowania (1 kHz = 1 ms)
period_s = 3.0  # Okres sinusa wynikający z omega (2*pi / omega = 3)
num_samples = int(sample_rate_hz * period_s)  # 3000 próbek

filename = "sine_lut.h"

with open(filename, "w") as f:
    f.write(f"// Wygenerowana tablica LUT dla sinusa\n")
    f.write(f"// Wzór: y = sin(t * 2pi/3) * 30 + 50\n")
    f.write(f"// Liczba próbek: {num_samples} (idealne dla pętli {sample_rate_hz} Hz)\n\n")
    f.write("#ifndef SINE_LUT_H\n#define SINE_LUT_H\n\n")
    f.write(f"const float sine_lut[{num_samples}] = {{\n    ")

    for i in range(num_samples):
        # Obliczenie aktualnego czasu 't' dla danej próbki
        t = i / sample_rate_hz
        val = math.sin(t * omega) * amplitude + offset

        # Zapis z dopiskiem 'f', aby STM32 traktował to jako float (FPU), a nie double
        f.write(f"{val:.3f}f, ")

        # Złamanie wiersza co 10 elementów dla czytelności
        if (i + 1) % 10 == 0:
            f.write("\n    ")

    f.write("\n};\n\n#endif // SINE_LUT_H\n")

print(f"Gotowe! Plik {filename} został wygenerowany. Zapisano {num_samples} elementów.")