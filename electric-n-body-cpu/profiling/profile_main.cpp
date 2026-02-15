// profile_main.cpp — Comprehensive profiling harness for the CPU N-Body simulation.
//
// Measures:
//   • Force computation time  (the O(N²) pairwise kernel)
//   • Integration time        (velocity + position update)
//   • VTK I/O time            (optional, disabled by default for pure compute benchmarks)
//   • Total step time
//
// Outputs CSV rows to stdout so they can be piped/appended to a results file.
//
// Usage:
//   ./nbody_profile <N> <steps> <threads> [--vtk]
//
// CSV columns:
//   n_particles, n_threads, n_steps,
//   avg_force_ms, avg_integrate_ms, avg_vtk_ms, avg_step_ms,
//   total_force_ms, total_integrate_ms, total_vtk_ms, total_sim_ms,
//   gflops_force, particles_per_sec

#include "particles.h"
#include "physics.h"
#include "simulation_config.h"
#include "vtk_writer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <sys/stat.h>
#include <vector>

// ── Timer helper ─────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>; // milliseconds

// ── Generate random particles (identical distribution to sample_generator.py) ──
static Particles generate_particles(size_t N)
{
    std::vector<Particle> raw(N);
    // deterministic seed for reproducibility
    std::srand(42);
    for (size_t i = 0; i < N; ++i)
    {
        auto randf = [](float lo, float hi) {
            return lo + static_cast<float>(std::rand()) / RAND_MAX * (hi - lo);
        };
        raw[i].x  = randf(-1.0f, 1.0f);
        raw[i].y  = randf(-1.0f, 1.0f);
        raw[i].z  = randf(-1.0f, 1.0f);
        raw[i].vx = 0.0f;
        raw[i].vy = 0.0f;
        raw[i].vz = 0.0f;
        raw[i].m  = 1.0f;
        raw[i].q  = randf(-1.6e-18f, 1.6e-18f);
    }
    return Particles(raw);
}

// ── Instrumented physics step ────────────────────────────────────
// Splits the physics_step into force + integrate, timing each.
// This mirrors physics.cpp exactly but with timing instrumentation.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float VACUUM_PERMITTIVITY = 8.8541878188e-12f;
static constexpr float VACUUM_PERMEABILITY = 1.25663706212e-6f;
static constexpr float K_E = 1.0f / (4.0f * static_cast<float>(M_PI) * VACUUM_PERMITTIVITY);
static constexpr float K_M = VACUUM_PERMEABILITY / (4.0f * static_cast<float>(M_PI));
static constexpr float SOFTENING = 1e-6f;
static constexpr float SOFT2 = SOFTENING * SOFTENING;

struct Vec3 { float x, y, z; };
static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(float s, Vec3 v) { return {s * v.x, s * v.y, s * v.z}; }
static inline float dot(Vec3 a, Vec3 b)       { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct StepTimings {
    double force_ms;
    double integrate_ms;
    double vtk_ms;
};

static StepTimings instrumented_physics_step(Particles &p, float dt, bool write_vtk,
                                              size_t step_idx)
{
    StepTimings t{};
    const size_t N = p.size();

    // ── Force computation ─────────────────────────────────────
    std::vector<float> fx(N, 0.0f), fy(N, 0.0f), fz(N, 0.0f);

    auto t0 = Clock::now();

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

            Vec3 r = ri - rj;
            float r2    = dot(r, r) + SOFT2;
            float inv_r = 1.0f / std::sqrt(r2);
            float inv_r3 = inv_r * inv_r * inv_r;

            float coeff_e = K_E * qj * qi * inv_r3;
            Vec3 F_electric = coeff_e * r;

            float coeff_m   = K_M * qj * qi * inv_r3;
            float vi_dot_r  = dot(vi, r);
            float vi_dot_vj = dot(vi, vj);
            Vec3 F_magnetic  = coeff_m * ((vi_dot_r * vj) - (vi_dot_vj * r));

            Vec3 F = F_electric + F_magnetic;
            sum_fx += F.x;
            sum_fy += F.y;
            sum_fz += F.z;
        }

        fx[i] = sum_fx;
        fy[i] = sum_fy;
        fz[i] = sum_fz;
    }

    auto t1 = Clock::now();
    t.force_ms = Duration(t1 - t0).count();

    // ── Integration ───────────────────────────────────────────
    auto t2 = Clock::now();

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

    auto t3 = Clock::now();
    t.integrate_ms = Duration(t3 - t2).count();

    // ── VTK I/O (optional) ────────────────────────────────────
    if (write_vtk)
    {
        auto t4 = Clock::now();
        std::ostringstream filename;
        filename << "vtk/profile_" << std::setfill('0') << std::setw(6) << step_idx << ".vtk";
        write_vtk_frame(p, filename.str());
        auto t5 = Clock::now();
        t.vtk_ms = Duration(t5 - t4).count();
    }

    return t;
}

// ── FLOPS estimation ─────────────────────────────────────────────
// Per pair interaction (counting only the j!=i body of the inner loop):
//   r = ri - rj:                    3 sub
//   dot(r,r) + SOFT2:               5 (3 mul + 1 add + 1 add)
//   1/sqrt:                         1 special (count as 1 FLOP)
//   inv_r3 = inv_r*inv_r*inv_r:     2 mul
//   coeff_e = K_E*qj*qi*inv_r3:     3 mul
//   F_electric = coeff_e * r:       3 mul
//   coeff_m = K_M*qj*qi*inv_r3:     3 mul
//   dot(vi,r):                      5 (3 mul + 2 add)
//   dot(vi,vj):                     5 (3 mul + 2 add)
//   vi_dot_r * vj:                  3 mul
//   vi_dot_vj * r:                  3 mul
//   subtract two vec3:              3 sub
//   coeff_m * (...):                3 mul
//   F = F_e + F_m:                  3 add
//   accumulate:                     3 add
//   ─────────────────────────────────
//   Total ≈ 45 FLOPS per pair
//
// Integration: ~12 FLOPS per particle
//
static constexpr double FLOPS_PER_PAIR = 45.0;
static constexpr double FLOPS_PER_INTEGRATE = 12.0;

static void print_csv_header()
{
    std::cout << "n_particles,n_threads,n_steps,"
              << "avg_force_ms,avg_integrate_ms,avg_vtk_ms,avg_step_ms,"
              << "total_force_ms,total_integrate_ms,total_vtk_ms,total_sim_ms,"
              << "gflops_force,particles_per_sec"
              << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <N_particles> <N_steps> <N_threads> [--vtk]\n";
        return 1;
    }

    size_t N          = std::atol(argv[1]);
    size_t num_steps  = std::atol(argv[2]);
    int    num_threads = std::atoi(argv[3]);
    bool   do_vtk     = (argc > 4 && std::string(argv[4]) == "--vtk");

    omp_set_num_threads(num_threads);

    // Print header only if this looks like a fresh run (N is small or first call)
    // Caller script handles headers
    if (std::getenv("PROFILE_PRINT_HEADER"))
        print_csv_header();

    // Generate particles
    Particles particles = generate_particles(N);

    if (do_vtk)
        mkdir("vtk", 0755);

    // ── Warm-up step (not counted) ───────────────────────────
    {
        Particles warmup = generate_particles(N);
        instrumented_physics_step(warmup, 0.001f, false, 0);
    }

    // ── Timed simulation ─────────────────────────────────────
    std::vector<double> force_times, integrate_times, vtk_times;
    force_times.reserve(num_steps);
    integrate_times.reserve(num_steps);
    vtk_times.reserve(num_steps);

    auto sim_start = Clock::now();

    for (size_t step = 0; step < num_steps; ++step)
    {
        auto timings = instrumented_physics_step(particles, 0.001f, do_vtk, step);
        force_times.push_back(timings.force_ms);
        integrate_times.push_back(timings.integrate_ms);
        vtk_times.push_back(timings.vtk_ms);
    }

    auto sim_end = Clock::now();
    double total_sim_ms = Duration(sim_end - sim_start).count();

    // ── Aggregate results ────────────────────────────────────
    auto sum = [](const std::vector<double> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    };
    auto avg = [&sum](const std::vector<double> &v) {
        return v.empty() ? 0.0 : sum(v) / static_cast<double>(v.size());
    };

    double total_force     = sum(force_times);
    double total_integrate = sum(integrate_times);
    double total_vtk       = sum(vtk_times);

    double avg_force     = avg(force_times);
    double avg_integrate = avg(integrate_times);
    double avg_vtk       = avg(vtk_times);
    double avg_step      = avg_force + avg_integrate + avg_vtk;

    // GFLOPS for force computation
    double total_pairs  = static_cast<double>(N) * (static_cast<double>(N) - 1.0) * num_steps;
    double total_flops_force = total_pairs * FLOPS_PER_PAIR;
    double gflops_force = total_flops_force / (total_force * 1e6); // ms $\rightarrow$ s, then / 1e9

    // Particles processed per second (interaction-throughput)
    double particles_per_sec = (static_cast<double>(N) * num_steps) / (total_sim_ms / 1000.0);

    // ── Output CSV row ───────────────────────────────────────
    std::cout << std::fixed << std::setprecision(4)
              << N << ","
              << num_threads << ","
              << num_steps << ","
              << avg_force << ","
              << avg_integrate << ","
              << avg_vtk << ","
              << avg_step << ","
              << total_force << ","
              << total_integrate << ","
              << total_vtk << ","
              << total_sim_ms << ","
              << std::setprecision(4) << gflops_force << ","
              << std::setprecision(0) << particles_per_sec
              << std::endl;

    // ── Human-readable summary to stderr ─────────────────────
    std::cerr << "\n══════════════════════════════════════════════════\n"
              << "  CPU N-Body Profiling Results\n"
              << "══════════════════════════════════════════════════\n"
              << "  Particles      : " << N << "\n"
              << "  Steps          : " << num_steps << "\n"
              << "  Threads        : " << num_threads << "\n"
              << "──────────────────────────────────────────────────\n"
              << "  Avg force      : " << std::fixed << std::setprecision(3) << avg_force << " ms\n"
              << "  Avg integrate  : " << avg_integrate << " ms\n"
              << "  Avg VTK I/O    : " << avg_vtk << " ms\n"
              << "  Avg step total : " << avg_step << " ms\n"
              << "──────────────────────────────────────────────────\n"
              << "  Total sim time : " << total_sim_ms << " ms\n"
              << "  Force GFLOPS   : " << std::setprecision(4) << gflops_force << "\n"
              << "  Particles/sec  : " << std::setprecision(0) << particles_per_sec << "\n"
              << "══════════════════════════════════════════════════\n";

    return 0;
}
