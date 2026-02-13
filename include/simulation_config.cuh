#pragma once

#include <cstddef>
#include <string>

enum class RendererType
{
    VTK,
    ROOT
};

struct SimulationConfig
{
    float dt{0.001f};
    size_t max_steps{1000};
    size_t output_interval{1};
    size_t status_interval{100};
    size_t queue_capacity{120};
    RendererType renderer{RendererType::VTK};
};
