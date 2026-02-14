#include "vtk_writer.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_vtk_frame(const Particles &particles, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t n = particles.size();

    file << "# vtk DataFile Version 3.0\n";
    file << "N-Body Simulation\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "POINTS " << n << " float\n";
    for (size_t i = 0; i < n; ++i)
    {
        file << std::scientific << std::setprecision(6)
             << particles.x[i] << " "
             << particles.y[i] << " "
             << particles.z[i] << "\n";
    }

    file << "\nPOINT_DATA " << n << "\n";
    file << "SCALARS charge float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n; ++i)
    {
        file << std::scientific << std::setprecision(6) << particles.q[i] << "\n";
    }
}
