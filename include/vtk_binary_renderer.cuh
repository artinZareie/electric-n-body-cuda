#pragma once

#include <renderer_interface.cuh>
#include <string>

class VtkBinaryRenderer : public IRenderer
{
  public:
    VtkBinaryRenderer() = default;
    ~VtkBinaryRenderer() override = default;

    void initialize(const std::string &output_path) override;
    void render(const Particles &particles, size_t frame_index) override;
    void shutdown() override;

  private:
    std::string m_output_path;
    bool m_initialized{false};
};

class VtkBinaryRendererFactory : public IRendererFactory
{
  public:
    std::unique_ptr<IRenderer> create() override;
};
