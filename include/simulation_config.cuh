#pragma once

#include <cstddef>

struct SimulationConfig
{
    float dt{0.001f};
    size_t max_steps{1000};
    size_t output_interval{1};
    size_t status_interval{100};
};
