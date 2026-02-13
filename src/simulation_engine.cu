#include <iostream>
#include <simulation_engine.cuh>

SimulationEngine::SimulationEngine(Particles &&particles, const SimulationConfig &config)
    : m_particles(std::move(particles)), m_config(config)
{
}

SimulationEngine::~SimulationEngine()
{
    stop();
    if (m_sim_thread.joinable())
        m_sim_thread.join();
}

void SimulationEngine::attach_renderer(std::unique_ptr<IRenderer> renderer)
{
    m_renderers.push_back(std::move(renderer));
}

void SimulationEngine::initialize()
{
    if (m_initialized)
        return;

    m_physics = std::make_unique<PhysicsSystem>(m_particles.size());
    m_particles.view();

    for (auto &renderer : m_renderers)
        renderer->initialize("vtk");

    m_initialized = true;
}

void SimulationEngine::step()
{
    auto view = m_particles.view();
    m_physics->step(view, m_config.dt);
}

void SimulationEngine::live_loop(FrameQueue &queue)
{
    std::cout << "Loaded " << m_particles.size() << " particles\n";
    std::cout << "Running live simulation...\n";

    size_t local_step = 0;
    while (m_running.load())
    {
        step();
        ++local_step;
        m_current_step.store(local_step);

        if (local_step % m_config.output_interval == 0)
        {
            m_particles.update_with_device();
            auto cpu = m_particles.cpu_view();
            size_t n = m_particles.size();

            Frame frame;
            frame.step = local_step;
            frame.x.assign(cpu.x, cpu.x + n);
            frame.y.assign(cpu.y, cpu.y + n);
            frame.z.assign(cpu.z, cpu.z + n);

            queue.push(std::move(frame));

            if (m_config.status_interval > 0 && local_step % m_config.status_interval == 0)
                std::cout << "Step " << local_step << "\n";
        }
    }

    queue.close();
    m_particles.update_and_unview();
    std::cout << "\nSimulation stopped at step " << local_step << "\n";
}

void SimulationEngine::batch_loop()
{
    std::cout << "Loaded " << m_particles.size() << " particles\n";
    std::cout << "Running simulation for " << m_config.max_steps << " steps...\n";

    for (size_t local_step = 0; m_running.load() && local_step <= m_config.max_steps; ++local_step)
    {
        if (local_step % m_config.output_interval == 0)
        {
            m_particles.update_with_device();
            for (auto &renderer : m_renderers)
                renderer->render(m_particles, local_step);

            if (m_config.status_interval > 0 && local_step % m_config.status_interval == 0)
                std::cout << "Step " << local_step << "/" << m_config.max_steps << "\n";
        }

        if (local_step < m_config.max_steps)
            step();

        m_current_step.store(local_step + 1);
    }

    for (auto &renderer : m_renderers)
        renderer->shutdown();

    m_particles.update_and_unview();
    m_running.store(false);
    std::cout << "\nSimulation complete!\n";
}

void SimulationEngine::run_async(FrameQueue &queue)
{
    initialize();
    m_running.store(true);
    m_current_step.store(0);
    m_sim_thread = std::thread(&SimulationEngine::live_loop, this, std::ref(queue));
}

void SimulationEngine::run_blocking()
{
    initialize();
    m_running.store(true);
    m_current_step.store(0);
    batch_loop();
}

void SimulationEngine::stop()
{
    m_running.store(false);
}
