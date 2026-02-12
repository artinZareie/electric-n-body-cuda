import numpy as np

N = int(input())

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = -1.0, 1.0
CHARGE_QUANTUM_COULOMB = 1.602176634e-19

Q_MIN, Q_MAX = -10 * CHARGE_QUANTUM_COULOMB, 10 * CHARGE_QUANTUM_COULOMB

OUTPUT_FILE = "particles.txt"

rng = np.random.default_rng()

x = rng.uniform(X_MIN, X_MAX, N)
y = rng.uniform(Y_MIN, Y_MAX, N)
z = rng.uniform(Z_MIN, Z_MAX, N)

vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)

m = np.ones(N)
q = rng.uniform(Q_MIN, Q_MAX, N)

with open(OUTPUT_FILE, "w") as f:
    f.write(f"{N}\n")
    for i in range(N):
        f.write(
            f"{x[i]:.8e} {y[i]:.8e} {z[i]:.8e} "
            f"{vx[i]:.8e} {vy[i]:.8e} {vz[i]:.8e} "
            f"{m[i]:.8e} {q[i]:.8e}\n"
        )

print(f"Wrote {N} particles to '{OUTPUT_FILE}'")
