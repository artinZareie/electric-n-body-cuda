#include <compute.cuh>
#include <physics_system.cuh>

#define USE_TILED_VERSION

PhysicsSystem::PhysicsSystem(size_t particle_count)
    : m_particle_count(particle_count), m_resources(std::make_unique<CudaResources>())
{
    cudaMalloc(&m_resources->net_forces, particle_count * sizeof(float4));
    
#ifndef USE_TILED_VERSION
    m_resources->forces_matrix = new Array2D<float4>();
    cudaMalloc(&m_resources->forces_matrix->data, particle_count * particle_count * sizeof(float4));
    m_resources->forces_matrix->nx = particle_count;
    m_resources->forces_matrix->ny = particle_count;
#endif
}

PhysicsSystem::~PhysicsSystem() = default;

void PhysicsSystem::step(ParticlesView &particles_view, float dt)
{
#ifdef USE_TILED_VERSION
    dim3 block_size(256);
    dim3 grid_size((m_particle_count + block_size.x - 1) / block_size.x);
    
    compute_force_tiled<<<grid_size, block_size>>>(particles_view, m_resources->net_forces, m_particle_count);
#else
    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d(
        (m_particle_count + block_size_2d.x - 1) / block_size_2d.x,
        (m_particle_count + block_size_2d.y - 1) / block_size_2d.y
    );
    
    compute_force<<<grid_size_2d, block_size_2d>>>(particles_view, *m_resources->forces_matrix, m_particle_count);
    
    dim3 reduction_block(256);
    dim3 reduction_grid(m_particle_count);
    size_t shared_mem_size = reduction_block.x * sizeof(float4);
    
    force_reduction<<<reduction_grid, reduction_block, shared_mem_size>>>(
        *m_resources->forces_matrix, m_resources->net_forces, m_particle_count);
#endif

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
    
    if (forces_matrix)
    {
        if (forces_matrix->data)
        {
            cudaFree(forces_matrix->data);
        }
        delete forces_matrix;
    }
}
