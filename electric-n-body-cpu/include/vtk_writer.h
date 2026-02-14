#pragma once

#include "particles.h"
#include <string>

void write_vtk_frame(const Particles &particles, const std::string &filename);
