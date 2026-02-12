#include <Array2D.cuh>
#include <compute.cuh>
#include <iomanip>
#include <iostream>
#include <particles.cuh>
#include <sys/stat.h>
#include <sys/types.h>
#include <vtk_writer.cuh>

int main()
{
    const float dt = 0.001f;
    const int num_steps = 1000;
    const int output_interval = 1;

    mkdir("vtk", 0755);

    std::cout << "Loading particles from file...\n";
    auto initiation = Particles::read_from_file("particles.txt");
    Particles particles(initiation);
    size_t N = particles.size();

    std::cout << "Loaded " << N << " particles\n";
    std::cout << "Running simulation for " << num_steps << " steps...\n";

    Array2D<float4> forces_matrix;
    forces_matrix.nx = N;
    forces_matrix.ny = N;
    cudaMalloc(&forces_matrix.data, N * N * sizeof(float4));
    float4 *d_net_forces;
    cudaMalloc(&d_net_forces, N * sizeof(float4));

    auto view = particles.view();

    for (int step = 0; step <= num_steps; ++step)
    {
        if (step % output_interval == 0)
        {
            std::ostringstream filename;
            filename << "vtk/output_" << std::setfill('0') << std::setw(6) << step << ".vtk";

            particles.update_with_device();
            write_vtk_frame(particles, filename.str());

            if (step % 100 == 0)
            {
                std::cout << "Step " << step << "/" << num_steps << " - Wrote " << filename.str() << "\n";
            }
        }

        if (step < num_steps)
        {
            dim3 block_size(16, 16);
            dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
            compute_force<<<grid_size, block_size>>>(view, forces_matrix, N);

            dim3 reduction_block(256);
            dim3 reduction_grid(N);
            size_t shared_mem_size = reduction_block.x * sizeof(float4);
            force_reduction<<<reduction_grid, reduction_block, shared_mem_size>>>(forces_matrix, d_net_forces, N);

            dim3 update_block(256);
            dim3 update_grid((N + update_block.x - 1) / update_block.x);
            update_particle_states<<<update_grid, update_block>>>(view, reinterpret_cast<float3 *>(d_net_forces), dt,
                                                                  N);

            cudaDeviceSynchronize();
        }
    }

    particles.update_and_unview();

    cudaFree(d_net_forces);
    cudaFree(forces_matrix.data);

    std::cout << "\nSimulation complete! VTK files written to vtk/ directory.\n";
    std::cout << "Run create_video_simple.py to generate video.\n";

    return 0;
}