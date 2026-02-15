#pragma once

#include <Array2D.cuh>
#include <cuda_runtime.h>
#include <memory>
#include <particles.cuh>

class PhysicsSystem
{
  public:
    PhysicsSystem(size_t particle_count);
    ~PhysicsSystem();

    PhysicsSystem(const PhysicsSystem &) = delete;
    PhysicsSystem &operator=(const PhysicsSystem &) = delete;
    PhysicsSystem(PhysicsSystem &&) = default;
    PhysicsSystem &operator=(PhysicsSystem &&) = default;

    void step(ParticlesView &particles_view, float dt);

  private:
    size_t m_particle_count;

    struct CudaResources
    {
        float4 *net_forces{nullptr};
        Array2D<float4> *forces_matrix{nullptr}; // Only used in non-tiled version

        ~CudaResources();
    };

    std::unique_ptr<CudaResources> m_resources;
};
