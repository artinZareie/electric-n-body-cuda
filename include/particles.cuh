#pragma once

#include "particle.cuh"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct ParticlesView
{
    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;
    float *__restrict__ m, *__restrict__ q;
};

class Particles
{
  private:
    std::vector<float> m_x, m_y, m_z;
    std::vector<float> m_vx, m_vy, m_vz;
    std::vector<float> m_m, m_q;

    struct DeviceMemory
    {
        float *x = nullptr, *y = nullptr, *z = nullptr;
        float *vx = nullptr, *vy = nullptr, *vz = nullptr;
        float *m = nullptr, *q = nullptr;

        ~DeviceMemory();
        DeviceMemory() = default;
        DeviceMemory(const DeviceMemory &) = delete;
        DeviceMemory &operator=(const DeviceMemory &) = delete;
        DeviceMemory(DeviceMemory &&) = delete;
        DeviceMemory &operator=(DeviceMemory &&) = delete;
    };

    std::unique_ptr<DeviceMemory> m_device_memory;

  public:
    static std::vector<Particle> read_from_file(const std::string &file_name);

  public:
    Particles(const std::vector<Particle> &);
    Particles(size_t n, float *x, float *y, float *z, float *vx, float *vy, float *vz, float *m, float *q);
    Particles(const Particles &);
    Particles(Particles &&) = default;
    Particles &operator=(const Particles &);
    Particles &operator=(Particles &&) = default;

    size_t size() const;

    ParticlesView view();
    void update_with_device();
    void update_and_unview();

    ParticlesView cpu_view();
    std::vector<Particle> as_vector() const;

    ~Particles() = default;
};