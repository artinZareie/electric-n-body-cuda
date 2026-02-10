#pragma once

#include "particle.hpp"
#include <cstddef>
#include <vector>

struct ParticlesView
{
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *m, *q;

  public:
    void cpu_free();
};

class Particles
{
  private:
    std::size_t m_size = 0;
    float *m_x, *m_y, *m_z;
    float *m_vx, *m_vy, *m_vz;
    float *m_m, *m_q;

    ParticlesView *m_view = nullptr;

  public:
    Particles(const std::vector<Particle> &);
    Particles(size_t n, float *x, float *y, float *z, float *vx, float *vy, float *vz, float *m, float *q);
    Particles(const Particles &);
    Particles(Particles &&);

    size_t size() const;

    ParticlesView view();
    void update_with_device();
    void update_and_unview();

    ParticlesView cpu_view();
    std::vector<Particle> as_vector();

    ~Particles();
};