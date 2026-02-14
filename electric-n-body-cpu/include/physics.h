#pragma once

#include "particles.h"
#include <cstddef>

// Compute all pairwise forces (Coulomb + magnetic) and update
// positions/velocities using Semi-Implicit Euler (matching the GPU version).
// The outer loop over particles is parallelised with OpenMP.
void physics_step(Particles &particles, float dt);
