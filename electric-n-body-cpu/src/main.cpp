// main.cpp  —  OpenMP CPU N-Body simulation
// Mirrors the GPU version's main.cu + simulation_engine logic.

#include "particles.h"
#include "physics.h"
#include "simulation_config.h"
#include "vtk_writer.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    const char *input_file = "particles.txt";
    if (argc > 1) input_file = argv[1];

    std::cout << "Loading particles from " << input_file << "...\n";
    auto raw_particles = Particles::read_from_file(input_file);
    Particles particles(raw_particles);

    SimulationConfig config;
    config.dt              = 0.001f;
    config.max_steps       = 1000;
    config.output_interval = 1;
    config.status_interval = 100;

    // Create vtk output directory
    mkdir("vtk", 0755);

    std::cout << "Loaded " << particles.size() << " particles\n";
    std::cout << "Running simulation for " << config.max_steps << " steps...\n";

    for (size_t step = 0; step <= config.max_steps; ++step)
    {
        // ── render (write VTK frame) ──
        if (step % config.output_interval == 0)
        {
            std::ostringstream filename;
            filename << "vtk/output_" << std::setfill('0') << std::setw(6) << step << ".vtk";
            write_vtk_frame(particles, filename.str());

            if (config.status_interval > 0 && step % config.status_interval == 0)
            {
                std::cout << "Step " << step << "/" << config.max_steps << "\n";
            }
        }

        // ── physics step ──
        if (step < config.max_steps)
        {
            physics_step(particles, config.dt);
        }
    }

    std::cout << "\nSimulation complete! VTK files written to vtk/ directory.\n";
    std::cout << "Run create_video_simple.py to generate video.\n";

    return 0;
}
