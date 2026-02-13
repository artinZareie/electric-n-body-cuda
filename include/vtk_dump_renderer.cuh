#pragma once

#include <renderer_interface.cuh>
#include <string>

class VtkRenderer : public IRenderer
{
  public:
    VtkRenderer() = default;
    ~VtkRenderer() override = default;

    void initialize(const std::string &output_path) override;
    void render(const Particles &particles, size_t frame_index) override;
    void shutdown() override;

  private:
    std::string m_output_path;
    bool m_initialized{false};
};

class VtkRendererFactory : public IRendererFactory
{
  public:
    std::unique_ptr<IRenderer> create() override;
};
