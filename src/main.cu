#include <cstring>
#include <frame_queue.cuh>
#include <iostream>
#include <particles.cuh>
#include <root_renderer.cuh>
#include <simulation_config.cuh>
#include <simulation_engine.cuh>
#include <vtk_renderer.cuh>

int main(int argc, char *argv[])
{
    RendererType renderer_type = RendererType::ROOT;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--root") == 0)
            renderer_type = RendererType::ROOT;
        else if (std::strcmp(argv[i], "--vtk") == 0)
            renderer_type = RendererType::VTK;
    }

    std::cout << "Loading particles from file...\n";
    auto initial_particles = Particles::read_from_file("particles.txt");

    SimulationConfig config{
        .dt = 0.001f,
        .max_steps = 1000,
        .output_interval = 1,
        .status_interval = 1000,
        .queue_capacity = 2,
        .renderer = renderer_type,
    };

    SimulationEngine engine(Particles{initial_particles}, config);

    if (renderer_type == RendererType::ROOT)
    {
        FrameQueue queue(config.queue_capacity);
        RootRenderer root;
        root.initialize("");

        engine.run_async(queue);
        root.run_event_loop(queue);
        engine.stop();
    }
    else
    {
        engine.attach_renderer(VtkRendererFactory{}.create());
        engine.run_blocking();
    }

    return 0;
}