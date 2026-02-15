#include <iostream>
#include <particles.cuh>
#include <simulation_config.cuh>
#include <simulation_engine.cuh>

#define USE_BINARY_VTK

#ifdef USE_BINARY_VTK
#include <vtk_binary_renderer.cuh>
#else
#include <vtk_renderer.cuh>
#endif

int main()
{
    std::cout << "Loading particles from file...\n";
    auto initial_particles = Particles::read_from_file("particles.txt");

    SimulationConfig config{.dt = 0.001f, .max_steps = 1000, .output_interval = 1, .status_interval = 100};

    SimulationEngine engine(Particles{initial_particles}, config);

#ifdef USE_BINARY_VTK
    VtkBinaryRendererFactory renderer_factory;
#else
    VtkRendererFactory renderer_factory;
#endif
    engine.attach_renderer(renderer_factory.create());

    engine.run();

    return 0;
}