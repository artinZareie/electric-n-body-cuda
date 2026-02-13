#pragma once

#include <memory>
#include <particles.cuh>
#include <string>

class IRenderer
{
  public:
    virtual ~IRenderer() = default;

    virtual void initialize(const std::string &output_path) = 0;
    virtual void render(const Particles &particles, size_t frame_index) = 0;
    virtual void shutdown() = 0;
};

class IRendererFactory
{
  public:
    virtual ~IRendererFactory() = default;
    virtual std::unique_ptr<IRenderer> create() = 0;
};
