#include "particles.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

Particles::Particles(const std::vector<Particle> &particles)
    : x(particles.size()), y(particles.size()), z(particles.size()),
      vx(particles.size()), vy(particles.size()), vz(particles.size()),
      m(particles.size()), q(particles.size())
{
    for (size_t i = 0; i < particles.size(); ++i)
    {
        x[i]  = particles[i].x;
        y[i]  = particles[i].y;
        z[i]  = particles[i].z;
        vx[i] = particles[i].vx;
        vy[i] = particles[i].vy;
        vz[i] = particles[i].vz;
        m[i]  = particles[i].m;
        q[i]  = particles[i].q;
    }
}

std::vector<Particle> Particles::read_from_file(const std::string &file_name)
{
    std::ifstream file(file_name);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open particle file: " + file_name);
    }

    std::vector<Particle> particles;

    size_t N;
    file >> N;

    float px, py, pz, pvx, pvy, pvz, pm, pq;
    for (size_t i = 0; i < N; ++i)
    {
        file >> px >> py >> pz >> pvx >> pvy >> pvz >> pm >> pq;
        particles.push_back({px, py, pz, pvx, pvy, pvz, pm, pq});
    }

    return particles;
}
