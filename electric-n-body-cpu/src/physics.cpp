// physics.cpp  —  OpenMP CPU implementation of the exact same force
// equations and integration scheme used in the CUDA version.
//
// Force model (from compute.cu):
//   F_electric  = K_E * qi * qj / (|r|^2 + eps^2)^(3/2) * r_vec
//   F_magnetic  = K_M * qi * qj / (|r|^2 + eps^2)^(3/2) * (dot(vi, r)*vj - dot(vi, vj)*r)
//
// Integration (from update_particle_states):
//   a  = F_net / m
//   v += a * dt          (Semi-Implicit Euler — velocity first)
//   x += v * dt

#include "physics.h"
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── physical constants (identical to compute.cu) ──────────────────
static constexpr float VACUUM_PERMITTIVITY = 8.8541878188e-12f;
static constexpr float VACUUM_PERMEABILITY = 1.25663706212e-6f;
static constexpr float K_E = 1.0f / (4.0f * static_cast<float>(M_PI) * VACUUM_PERMITTIVITY);
static constexpr float K_M = VACUUM_PERMEABILITY / (4.0f * static_cast<float>(M_PI));
static constexpr float SOFTENING = 1e-6f;
static constexpr float SOFT2 = SOFTENING * SOFTENING;

// ── helpers ───────────────────────────────────────────────────────
struct Vec3 { float x, y, z; };

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(float s, Vec3 v) { return {s * v.x, s * v.y, s * v.z}; }
static inline float dot(Vec3 a, Vec3 b)       { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ── physics step ──────────────────────────────────────────────────
void physics_step(Particles &p, float dt)
{
    const size_t N = p.size();

    // Accumulate net force per particle (thread-private).
    std::vector<float> fx(N, 0.0f), fy(N, 0.0f), fz(N, 0.0f);

    // ---------- pairwise force computation (O(N^2)) ---------------
    // The GPU version stores the full N×N matrix then reduces rows.
    // Here we compute the same per-row sums directly with OpenMP.
    //
    // NOTE: the GPU kernel computes F[y][x] as the force ON particle y
    //       DUE TO particle x.  We replicate that exactly below.

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < N; ++i)
    {
        Vec3 ri  = {p.x[i],  p.y[i],  p.z[i]};
        Vec3 vi  = {p.vx[i], p.vy[i], p.vz[i]};
        float qi = p.q[i];

        float sum_fx = 0.0f, sum_fy = 0.0f, sum_fz = 0.0f;

        for (size_t j = 0; j < N; ++j)
        {
            if (j == i) continue;

            Vec3 rj  = {p.x[j],  p.y[j],  p.z[j]};
            Vec3 vj  = {p.vx[j], p.vy[j], p.vz[j]};
            float qj = p.q[j];

            // r = ri - rj  (matches GPU: ry - rx where y=i, x=j)
            Vec3 r = ri - rj;

            float r2    = dot(r, r) + SOFT2;
            float inv_r = 1.0f / std::sqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;

            // ── Coulomb (electric) force ──
            float coeff_e = K_E * qj * qi * inv_r3;
            Vec3 F_electric = coeff_e * r;

            // ── Magnetic force ──
            float coeff_m   = K_M * qj * qi * inv_r3;
            float vi_dot_r  = dot(vi, r);
            float vi_dot_vj = dot(vi, vj);
            Vec3 F_magnetic  = coeff_m * ((vi_dot_r * vj) - (vi_dot_vj * r));
            // Breakdown:  coeff_m * ( dot(vi,r)*vj  -  dot(vi,vj)*r )
            //   note: uses   vi_dot_r * vj   as a Vec3 scalar*vec
            //   and          vi_dot_vj * r    as a Vec3 scalar*vec

            // This matches GPU line:
            //   coeff_m * (vy_dot_r * vx  -  vy_dot_vx * r)
            // where vy=vi (particle being computed) and vx=vj (partner).

            Vec3 F = F_electric + F_magnetic;

            sum_fx += F.x;
            sum_fy += F.y;
            sum_fz += F.z;
        }

        fx[i] = sum_fx;
        fy[i] = sum_fy;
        fz[i] = sum_fz;
    }

    // ---------- integration (Semi-Implicit Euler) -----------------
    // Same as update_particle_states in compute.cu:
    //   v += (F/m) * dt
    //   x += v * dt  (uses updated v)

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
    {
        float inv_m = 1.0f / p.m[i];
        float ax = fx[i] * inv_m;
        float ay = fy[i] * inv_m;
        float az = fz[i] * inv_m;

        p.vx[i] += ax * dt;
        p.vy[i] += ay * dt;
        p.vz[i] += az * dt;

        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }
}
