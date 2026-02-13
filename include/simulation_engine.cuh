#pragma once

#include <functional>
#include <memory>
#include <particles.cuh>
#include <physics_system.cuh>
#include <renderer_interface.cuh>
#include <simulation_config.cuh>
#include <vector>

class SimulationEngine
{
  public:
    SimulationEngine(Particles &&particles, const SimulationConfig &config);
    ~SimulationEngine() = default;

    SimulationEngine(const SimulationEngine &) = delete;
    SimulationEngine &operator=(const SimulationEngine &) = delete;
    SimulationEngine(SimulationEngine &&) = default;
    SimulationEngine &operator=(SimulationEngine &&) = default;

    void attach_renderer(std::unique_ptr<IRenderer> renderer);
    void on_step_completed(std::function<void(size_t)> callback);

    void run();
    void stop();

    size_t current_step() const
    {
        return m_current_step;
    }
    bool is_running() const
    {
        return m_running;
    }
    const Particles &particles() const
    {
        return m_particles;
    }

  private:
    void initialize();
    void step();
    void render();
    void cleanup();

    Particles m_particles;
    SimulationConfig m_config;
    std::unique_ptr<PhysicsSystem> m_physics;
    std::vector<std::unique_ptr<IRenderer>> m_renderers;
    std::vector<std::function<void(size_t)>> m_step_callbacks;

    size_t m_current_step{0};
    bool m_running{false};
    bool m_initialized{false};
};
