#include <iostream>
#include <simulation_engine.cuh>

SimulationEngine::SimulationEngine(Particles &&particles, const SimulationConfig &config)
    : m_particles(std::move(particles)), m_config(config)
{
}

void SimulationEngine::attach_renderer(std::unique_ptr<IRenderer> renderer)
{
    m_renderers.push_back(std::move(renderer));
}

void SimulationEngine::on_step_completed(std::function<void(size_t)> callback)
{
    m_step_callbacks.push_back(std::move(callback));
}

void SimulationEngine::initialize()
{
    if (m_initialized)
        return;

    m_physics = std::make_unique<PhysicsSystem>(m_particles.size());
    m_particles.view();

    for (auto &renderer : m_renderers)
    {
        renderer->initialize("vtk");
    }

    m_initialized = true;
}

void SimulationEngine::step()
{
    auto view = m_particles.view();
    m_physics->step(view, m_config.dt);

    for (const auto &callback : m_step_callbacks)
    {
        callback(m_current_step);
    }
}

void SimulationEngine::render()
{
    m_particles.update_with_device();

    for (auto &renderer : m_renderers)
    {
        renderer->render(m_particles, m_current_step);
    }
}

void SimulationEngine::cleanup()
{
    for (auto &renderer : m_renderers)
    {
        renderer->shutdown();
    }

    m_particles.update_and_unview();
}

void SimulationEngine::run()
{
    initialize();

    m_running = true;
    m_current_step = 0;

    std::cout << "Loaded " << m_particles.size() << " particles\n";
    std::cout << "Running simulation for " << m_config.max_steps << " steps...\n";

    while (m_running && m_current_step <= m_config.max_steps)
    {
        if (m_current_step % m_config.output_interval == 0)
        {
            render();

            if (m_config.status_interval > 0 && m_current_step % m_config.status_interval == 0)
            {
                std::cout << "Step " << m_current_step << "/" << m_config.max_steps << "\n";
            }
        }

        if (m_current_step < m_config.max_steps)
        {
            step();
        }

        ++m_current_step;
    }

    cleanup();
    m_running = false;

    std::cout << "\nSimulation complete! VTK files written to vtk/ directory.\n";
    std::cout << "Run create_video_simple.py to generate video.\n";
}

void SimulationEngine::stop()
{
    m_running = false;
}
