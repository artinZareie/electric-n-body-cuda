#include "Array2D.cuh"
#include "compute.cuh"
#include "particles.cuh"
#include <cmath>
#include <cuda_runtime.h>

constexpr float VACUUM_PERMITTIVITY = 8.8541878188e-12f;
constexpr float VACUUM_PERMEABILITY = 1.25663706212e-6f;
constexpr float K_E = 1.0f / (4.0f * M_PI * VACUUM_PERMITTIVITY);
constexpr float K_M = VACUUM_PERMEABILITY / (4.0f * M_PI);
constexpr float SOFTENING = 1e-6f;
constexpr float SOFT2 = SOFTENING * SOFTENING;

__device__ inline float3 operator*(float a, float3 v)
{
    return make_float3(a * v.x, a * v.y, a * v.z);
}

__device__ inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void compute_force(const ParticlesView particles, Array2D<float4> forces_matrix, size_t N)
{
    const size_t tx = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t ty = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t stride_x = blockDim.x * gridDim.x;
    const size_t stride_y = blockDim.y * gridDim.y;

    for (size_t y = ty; y < N; y += stride_y)
    {
        const float3 ry = make_float3(particles.x[y], particles.y[y], particles.z[y]);
        const float3 vy = make_float3(particles.vx[y], particles.vy[y], particles.vz[y]);
        const float qy = particles.q[y];

        for (size_t x = tx; x < N; x += stride_x)
        {
            if (x == y)
            {
                forces_matrix(y, x) = make_float4(0.f, 0.f, 0.f, 0.f);
                continue;
            }

            const float3 rx = make_float3(particles.x[x], particles.y[x], particles.z[x]);
            const float3 vx = make_float3(particles.vx[x], particles.vy[x], particles.vz[x]);
            const float qx = particles.q[x];

            float3 r = make_float3(ry.x - rx.x, ry.y - rx.y, ry.z - rx.z);

            float r2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFT2;

            float inv_r = rsqrtf(r2);
            float inv_r3 = inv_r * inv_r * inv_r;

            float coeff_e = K_E * qx * qy * inv_r3;
            float3 F_electric = coeff_e * r;

            float coeff_m = K_M * qx * qy * inv_r3;
            float vy_dot_r = dot(vy, r);
            float vy_dot_vx = dot(vy, vx);
            float3 F_magnetic = coeff_m * (vy_dot_r * vx - vy_dot_vx * r);

            float3 F = F_electric + F_magnetic;

            forces_matrix(y, x) = make_float4(F.x, F.y, F.z, 0.0f);
        }
    }
}

__global__ void force_reduction(const Array2D<float4> forces_matrix, float4 *net_forces, size_t N)
{
    extern __shared__ float4 sdata[];

    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t block_size = blockDim.x;

    if (row >= N)
        return;

    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t j = tid; j < N; j += block_size)
    {
        float4 F = forces_matrix(row, j);
        sum.x += F.x;
        sum.y += F.y;
        sum.z += F.z;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = block_size / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid].x += sdata[tid + s].x;
            sdata[tid].y += sdata[tid + s].y;
            sdata[tid].z += sdata[tid + s].z;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        net_forces[row] = sdata[0];
    }
}

__global__ void update_particle_states(ParticlesView particles, const float3 *net_forces, float dt, size_t N)
{
    const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t num_threads = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += num_threads)
    {
        float3 F = net_forces[i];
        float m = particles.m[i];

        float3 a = make_float3(F.x / m, F.y / m, F.z / m);

        particles.vx[i] += a.x * dt;
        particles.vy[i] += a.y * dt;
        particles.vz[i] += a.z * dt;

        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
}