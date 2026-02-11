#pragma once

#include "Array2D.cuh"
#include "particles.cuh"
#include <cuda_runtime.h>

__global__ void compute_force(const ParticlesView particles, Array2D<float4> forces_matrix, size_t N);

__global__ void force_reduction(const Array2D<float4> forces_matrix, float4 *net_forces, size_t N);

__global__ void update_particle_states(ParticlesView particles, const float3 *net_forces, float dt, size_t N);
