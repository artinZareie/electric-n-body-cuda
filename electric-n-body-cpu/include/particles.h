#pragma once

#include "particle.h"
#include <cstddef>
#include <string>
#include <vector>

// SoA container — mirrors the GPU version's ParticlesView layout
// but lives entirely on the CPU.
class Particles
{
  public:
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> m, q;

    Particles() = default;
    explicit Particles(const std::vector<Particle> &particles);

    size_t size() const { return x.size(); }

    static std::vector<Particle> read_from_file(const std::string &file_name);
};
