#include "Array2D.hpp"
#include "particles.hpp"
#include <cmath>
#include <cuda_runtime.h>

constexpr float VACUUM_PERMITTIVITY = 8.8541878188e-12f;
constexpr float K_E = 1.0f / (4.0f * M_PI * VACUUM_PERMITTIVITY);
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

__global__ void compute_force(const ParticlesView particles, Array2D<float4> forces_matrix, size_t N)
{
    const size_t tx = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t ty = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t stride_x = blockDim.x * gridDim.x;
    const size_t stride_y = blockDim.y * gridDim.y;

    for (size_t y = ty; y < N; y += stride_y)
    {
        const float3 ry = make_float3(particles.x[y], particles.y[y], particles.z[y]);
        const float qy = particles.q[y];

        for (size_t x = tx; x < N; x += stride_x)
        {
            if (x == y)
            {
                forces_matrix(y, x) = make_float4(0.f, 0.f, 0.f, 0.f);
                continue;
            }

            const float3 rx = make_float3(particles.x[x], particles.y[x], particles.z[x]);
            const float qx = particles.q[x];

            float3 r = make_float3(ry.x - rx.x, ry.y - rx.y, ry.z - rx.z);

            float r2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFT2;

            float inv_r = rsqrtf(r2);
            float inv_r3 = inv_r * inv_r * inv_r;

            float coeff = K_E * qx * qy * inv_r3;

            float3 F = coeff * r;

            forces_matrix(y, x) = make_float4(F.x, F.y, F.z, 0.0f);
        }
    }
}
