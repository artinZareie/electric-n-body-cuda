#pragma once

#include "particles.cuh"
#include <string>

void write_vtk_frame_binary(const Particles &particles, const std::string &filename);
