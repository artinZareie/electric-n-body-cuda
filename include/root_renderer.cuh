#pragma once

#include <frame_queue.cuh>
#include <renderer_interface.cuh>

class TApplication;
class TCanvas;
class TPolyMarker3D;
class TTimer;

class RootRenderer : public IRenderer
{
  public:
    RootRenderer() = default;
    ~RootRenderer() override;

    void initialize(const std::string &output_path) override;
    void render(const Particles &particles, size_t frame_index) override;
    void shutdown() override;
    bool is_interactive() const override
    {
        return true;
    }

    void run_event_loop(FrameQueue &queue);
    void display_frame(const Frame &frame);

  private:
    TApplication *m_app{nullptr};
    TCanvas *m_canvas{nullptr};
    TPolyMarker3D *m_marker{nullptr};
    TTimer *m_timer{nullptr};
    bool m_initialized{false};
};
