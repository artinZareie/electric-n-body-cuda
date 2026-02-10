#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <particle.hpp>
#include <particles.hpp>
#include <stdexcept>
#include <vector>

Particles::Particles(const std::vector<Particle> &particles)
    : m_x(new float[particles.size()]), m_y(new float[particles.size()]), m_z(new float[particles.size()]),
      m_vx(new float[particles.size()]), m_vy(new float[particles.size()]), m_vz(new float[particles.size()]),
      m_m(new float[particles.size()]), m_q(new float[particles.size()]), m_size(particles.size())
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
    : m_x(new float[n]), m_y(new float[n]), m_z(new float[n]), m_vx(new float[n]), m_vy(new float[n]),
      m_vz(new float[n]), m_m(new float[n]), m_q(new float[n]), m_size(n)
{
    std::memcpy(m_x, x, n * sizeof(float));
    std::memcpy(m_y, y, n * sizeof(float));
    std::memcpy(m_z, z, n * sizeof(float));

    std::memcpy(m_vx, vx, n * sizeof(float));
    std::memcpy(m_vy, vy, n * sizeof(float));
    std::memcpy(m_vz, vz, n * sizeof(float));

    std::memcpy(m_m, m, n * sizeof(float));
    std::memcpy(m_q, q, n * sizeof(float));
}

Particles::Particles(const Particles &other)
    : m_x(new float[other.m_size]), m_y(new float[other.m_size]), m_z(new float[other.m_size]),
      m_vx(new float[other.m_size]), m_vy(new float[other.m_size]), m_vz(new float[other.m_size]),
      m_m(new float[other.m_size]), m_q(new float[other.m_size]), m_size(other.m_size)
{
    const size_t n = other.m_size;
    std::memcpy(m_x, other.m_x, n * sizeof(float));
    std::memcpy(m_y, other.m_y, n * sizeof(float));
    std::memcpy(m_z, other.m_z, n * sizeof(float));

    std::memcpy(m_vx, other.m_vx, n * sizeof(float));
    std::memcpy(m_vy, other.m_vy, n * sizeof(float));
    std::memcpy(m_vz, other.m_vz, n * sizeof(float));

    std::memcpy(m_m, other.m_m, n * sizeof(float));
    std::memcpy(m_q, other.m_q, n * sizeof(float));
}

Particles::Particles(Particles &&other)
    : m_x(other.m_x), m_y(other.m_y), m_z(other.m_z), m_vx(other.m_vx), m_vy(other.m_vy), m_vz(other.m_vz),
      m_m(other.m_m), m_q(other.m_q), m_size(other.m_size)
{
    other.m_size = 0;
    other.m_x = nullptr;
    other.m_y = nullptr;
    other.m_z = nullptr;
    other.m_vx = nullptr;
    other.m_vy = nullptr;
    other.m_vz = nullptr;
    other.m_m = nullptr;
    other.m_q = nullptr;
}

size_t Particles::size() const
{
    return m_size;
}

ParticlesView Particles::view()
{
    if (!m_view)
    {
        m_view = new ParticlesView;

        cudaMalloc(&m_view->x, m_size * sizeof(float));
        cudaMalloc(&m_view->y, m_size * sizeof(float));
        cudaMalloc(&m_view->z, m_size * sizeof(float));

        cudaMalloc(&m_view->vx, m_size * sizeof(float));
        cudaMalloc(&m_view->vy, m_size * sizeof(float));
        cudaMalloc(&m_view->vz, m_size * sizeof(float));

        cudaMalloc(&m_view->m, m_size * sizeof(float));
        cudaMalloc(&m_view->q, m_size * sizeof(float));

        cudaMemcpy(m_view->x, m_x, m_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_view->y, m_y, m_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_view->z, m_z, m_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_view->vx, m_vx, m_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_view->vy, m_vy, m_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_view->vz, m_vz, m_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_view->m, m_m, m_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_view->q, m_q, m_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    return *m_view;
}

void Particles::update_with_device()
{
    if (!m_view)
        throw std::runtime_error("No views found!");

    cudaMemcpy(m_x, m_view->x, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_y, m_view->y, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_z, m_view->z, m_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_vx, m_view->vx, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_vy, m_view->vy, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_vz, m_view->vz, m_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_m, m_view->m, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_q, m_view->q, m_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Particles::update_and_unview()
{
    update_with_device();

    cudaFree(m_view->x);
    cudaFree(m_view->y);
    cudaFree(m_view->z);

    cudaFree(m_view->vx);
    cudaFree(m_view->vy);
    cudaFree(m_view->vz);

    cudaFree(m_view->m);
    cudaFree(m_view->q);

    delete m_view;
    m_view = nullptr;
}

Particles::~Particles()
{
    if (m_size != 0)
    {
        if (m_x != nullptr)
            delete[] m_x;

        if (m_y != nullptr)
            delete[] m_y;

        if (m_z != nullptr)
            delete[] m_z;

        if (m_vx != nullptr)
            delete[] m_vx;

        if (m_vy != nullptr)
            delete[] m_vy;

        if (m_vz != nullptr)
            delete[] m_vz;

        if (m_m != nullptr)
            delete[] m_m;

        if (m_q != nullptr)
            delete[] m_q;
    }

    if (m_view)
    {
        cudaFree(m_view->x);
        cudaFree(m_view->y);
        cudaFree(m_view->z);
        cudaFree(m_view->vx);
        cudaFree(m_view->vy);
        cudaFree(m_view->vz);
        cudaFree(m_view->m);
        cudaFree(m_view->q);

        delete m_view;
    }
}

void ParticlesView::cpu_free()
{
    if (x)
    {
        delete[] x;
    }

    if (y)
    {
        delete[] y;
    }

    if (z)
    {
        delete[] z;
    }

    if (vx)
    {
        delete[] vx;
    }

    if (vy)
    {
        delete[] vy;
    }

    if (vz)
    {
        delete[] vz;
    }

    if (m)
    {
        delete[] m;
    }

    if (q)
    {
        delete[] q;
    }

    x = y = z = nullptr;
    vx = vy = vz = nullptr;
    m = q = nullptr;
}

ParticlesView Particles::cpu_view()
{
    return ParticlesView{m_x, m_y, m_z, m_vx, m_vy, m_vz, m_m, m_q};
}

std::vector<Particle> Particles::as_vector()
{
    std::vector<Particle> particles;

    for (size_t i = 0; i < m_size; i++)
    {
        particles.emplace_back(m_x[i], m_y[i], m_z[i], m_vx[i], m_vy[i], m_vz[i], m_m[i], m_q[i]);
    }

    return particles;
}