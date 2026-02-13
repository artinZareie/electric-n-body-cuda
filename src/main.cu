#include <iostream>
#include <particles.cuh>
#include <simulation_config.cuh>
#include <simulation_engine.cuh>
#include <vtk_renderer.cuh>

int main()
{
    std::cout << "Loading particles from file...\n";
    auto initial_particles = Particles::read_from_file("particles.txt");

    SimulationConfig config{.dt = 0.001f, .max_steps = 1000, .output_interval = 1, .status_interval = 100};

    SimulationEngine engine(Particles{initial_particles}, config);

    VtkRendererFactory renderer_factory;
    engine.attach_renderer(renderer_factory.create());

    engine.run();

    return 0;
}