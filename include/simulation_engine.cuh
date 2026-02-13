#pragma once

#include <atomic>
#include <frame_queue.cuh>
#include <memory>
#include <particles.cuh>
#include <physics_system.cuh>
#include <renderer_interface.cuh>
#include <simulation_config.cuh>
#include <thread>
#include <vector>

class SimulationEngine
{
  public:
    SimulationEngine(Particles &&particles, const SimulationConfig &config);
    ~SimulationEngine();

    SimulationEngine(const SimulationEngine &) = delete;
    SimulationEngine &operator=(const SimulationEngine &) = delete;
    SimulationEngine(SimulationEngine &&) = delete;
    SimulationEngine &operator=(SimulationEngine &&) = delete;

    void attach_renderer(std::unique_ptr<IRenderer> renderer);
    void run_async(FrameQueue &queue);
    void run_blocking();
    void stop();

    bool is_running() const
    {
        return m_running.load();
    }

  private:
    void live_loop(FrameQueue &queue);
    void batch_loop();
    void initialize();
    void step();

    Particles m_particles;
    SimulationConfig m_config;
    std::unique_ptr<PhysicsSystem> m_physics;
    std::vector<std::unique_ptr<IRenderer>> m_renderers;

    std::atomic<size_t> m_current_step{0};
    std::atomic<bool> m_running{false};
    bool m_initialized{false};

    std::thread m_sim_thread;
};
