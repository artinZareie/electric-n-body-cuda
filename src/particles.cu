#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <particle.cuh>
#include <particles.cuh>
#include <stdexcept>
#include <vector>

Particles::Particles(const std::vector<Particle> &particles)
    : m_x(particles.size()), m_y(particles.size()), m_z(particles.size()), m_vx(particles.size()),
      m_vy(particles.size()), m_vz(particles.size()), m_m(particles.size()), m_q(particles.size())
{
    for (size_t i = 0; i < particles.size(); i++)
    {
        m_x[i] = particles[i].x;
        m_y[i] = particles[i].y;
        m_z[i] = particles[i].z;
        m_vx[i] = particles[i].vx;
        m_vy[i] = particles[i].vy;
        m_vz[i] = particles[i].vz;
        m_m[i] = particles[i].m;
        m_q[i] = particles[i].q;
    }
}

Particles::Particles(size_t n, float *x, float *y, float *z, float *vx, float *vy, float *vz, float *m, float *q)
    : m_x(x, x + n), m_y(y, y + n), m_z(z, z + n), m_vx(vx, vx + n), m_vy(vy, vy + n), m_vz(vz, vz + n), m_m(m, m + n),
      m_q(q, q + n)
{
}

Particles::Particles(const Particles &other)
    : m_x(other.m_x), m_y(other.m_y), m_z(other.m_z), m_vx(other.m_vx), m_vy(other.m_vy), m_vz(other.m_vz),
      m_m(other.m_m), m_q(other.m_q)
{
}

Particles &Particles::operator=(const Particles &other)
{
    if (this != &other)
    {
        m_x = other.m_x;
        m_y = other.m_y;
        m_z = other.m_z;
        m_vx = other.m_vx;
        m_vy = other.m_vy;
        m_vz = other.m_vz;
        m_m = other.m_m;
        m_q = other.m_q;
        m_device_memory.reset();
    }
    return *this;
}

size_t Particles::size() const
{
    return m_x.size();
}

ParticlesView Particles::view()
{
    if (!m_device_memory)
    {
        m_device_memory = std::make_unique<DeviceMemory>();
        const size_t n = size();

        cudaMalloc(&m_device_memory->x, n * sizeof(float));
        cudaMalloc(&m_device_memory->y, n * sizeof(float));
        cudaMalloc(&m_device_memory->z, n * sizeof(float));

        cudaMalloc(&m_device_memory->vx, n * sizeof(float));
        cudaMalloc(&m_device_memory->vy, n * sizeof(float));
        cudaMalloc(&m_device_memory->vz, n * sizeof(float));

        cudaMalloc(&m_device_memory->m, n * sizeof(float));
        cudaMalloc(&m_device_memory->q, n * sizeof(float));

        cudaMemcpy(m_device_memory->x, m_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_memory->y, m_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_memory->z, m_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_device_memory->vx, m_vx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_memory->vy, m_vy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_memory->vz, m_vz.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_device_memory->m, m_m.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_memory->q, m_q.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }

    return ParticlesView{m_device_memory->x,  m_device_memory->y,  m_device_memory->z, m_device_memory->vx,
                         m_device_memory->vy, m_device_memory->vz, m_device_memory->m, m_device_memory->q};
}

void Particles::update_with_device()
{
    if (!m_device_memory)
        throw std::runtime_error("No device memory allocated!");

    const size_t n = size();
    cudaMemcpy(m_x.data(), m_device_memory->x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_y.data(), m_device_memory->y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_z.data(), m_device_memory->z, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_vx.data(), m_device_memory->vx, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_vy.data(), m_device_memory->vy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_vz.data(), m_device_memory->vz, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_m.data(), m_device_memory->m, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_q.data(), m_device_memory->q, n * sizeof(float), cudaMemcpyDeviceToHost);
}

void Particles::update_and_unview()
{
    update_with_device();
    m_device_memory.reset();
}

Particles::DeviceMemory::~DeviceMemory()
{
    if (x)
        cudaFree(x);
    if (y)
        cudaFree(y);
    if (z)
        cudaFree(z);
    if (vx)
        cudaFree(vx);
    if (vy)
        cudaFree(vy);
    if (vz)
        cudaFree(vz);
    if (m)
        cudaFree(m);
    if (q)
        cudaFree(q);
}

ParticlesView Particles::cpu_view()
{
    return ParticlesView{m_x.data(),  m_y.data(),  m_z.data(), m_vx.data(),
                         m_vy.data(), m_vz.data(), m_m.data(), m_q.data()};
}

std::vector<Particle> Particles::as_vector() const
{
    std::vector<Particle> particles;
    const size_t n = size();

    for (size_t i = 0; i < n; i++)
    {
        particles.emplace_back(m_x[i], m_y[i], m_z[i], m_vx[i], m_vy[i], m_vz[i], m_m[i], m_q[i]);
    }

    return particles;
}

std::vector<Particle> Particles::read_from_file(const std::string &file_name)
{
    std::ifstream file(file_name);
    std::vector<Particle> particles;

    size_t N;
    float x, y, z, vx, vy, vz, m, q;

    file >> N;
    for (size_t i = 0; i < N; i++)
    {
        file >> x >> y >> z >> vx >> vy >> vz >> m >> q;
        particles.emplace_back(x, y, z, vx, vy, vz, m, q);
    }

    return particles;
}