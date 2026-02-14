# N-Body Simulation — OpenMP CPU Version

An OpenMP-parallelised CPU implementation of the same electrostatic + magnetic
N-body simulation found in `electric-n-body-cuda`.  Produces **identical VTK
output** so the same ParaView / gnuplot / video scripts work unchanged.

## Physics (identical to GPU version)

| Component | Formula |
|-----------|---------|
| Coulomb   | `F_e = K_E · qi·qj / (r² + ε²)^(3/2) · r̂` |
| Magnetic  | `F_m = K_M · qi·qj / (r² + ε²)^(3/2) · (dot(vi,r)·vj − dot(vi,vj)·r)` |
| Integration | Semi-Implicit Euler: `v += a·dt`, `x += v·dt` |

## Build

```bash
make            # produces ./nbody_cpu
make clean
```

Requires `g++` with OpenMP support (`-fopenmp`).

## Run

```bash
# Use the same particles.txt from the CUDA project
cp ../electric-n-body-cuda/particles.txt .
./nbody_cpu                  # reads particles.txt, writes vtk/
./nbody_cpu my_particles.txt # custom input

# Control threads:
OMP_NUM_THREADS=8 ./nbody_cpu
```

## Output

VTK files are written to `vtk/` in the exact same legacy ASCII format
(POLYDATA, SCALARS charge) as the CUDA version — loadable in ParaView/VisIt.

## File structure

```
electric-n-body-cpu/
├── Makefile
├── README.md
├── include/
│   ├── particle.h
│   ├── particles.h
│   ├── physics.h
│   ├── simulation_config.h
│   └── vtk_writer.h
└── src/
    ├── main.cpp
    ├── particles.cpp
    ├── physics.cpp
    └── vtk_writer.cpp
```
