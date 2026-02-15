#include <compute.cuh>
#include <physics_system.cuh>

PhysicsSystem::PhysicsSystem(size_t particle_count)
    : m_particle_count(particle_count), m_resources(std::make_unique<CudaResources>())
{
    cudaMalloc(&m_resources->net_forces, particle_count * sizeof(float4));
}

PhysicsSystem::~PhysicsSystem() = default;

void PhysicsSystem::step(ParticlesView &particles_view, float dt)
{
    dim3 block_size(256);
    dim3 grid_size((m_particle_count + block_size.x - 1) / block_size.x);

    compute_force_tiled<<<grid_size, block_size>>>(particles_view, m_resources->net_forces, m_particle_count);

    dim3 update_block(256);
    dim3 update_grid((m_particle_count + update_block.x - 1) / update_block.x);

    update_particle_states<<<update_grid, update_block>>>(
        particles_view, reinterpret_cast<float3 *>(m_resources->net_forces), dt, m_particle_count);

    cudaDeviceSynchronize();
}

PhysicsSystem::CudaResources::~CudaResources()
{
    if (net_forces)
    {
        cudaFree(net_forces);
    }
}
