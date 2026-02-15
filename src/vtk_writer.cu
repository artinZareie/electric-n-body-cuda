#include "vtk_writer.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

static constexpr float MIN_INTENSITY = 0.15f;

void write_vtk_frame(const Particles &particles, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    auto cpu_view = const_cast<Particles &>(particles).cpu_view();
    size_t n = particles.size();

    // Find max absolute charge for normalization
    float max_abs_q = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        max_abs_q = std::max(max_abs_q, std::fabs(cpu_view.q[i]));
    }
    if (max_abs_q == 0.0f)
        max_abs_q = 1.0f;

    file << "# vtk DataFile Version 3.0\n";
    file << "N-Body Simulation\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "POINTS " << n << " float\n";
    for (size_t i = 0; i < n; ++i)
    {
        file << std::scientific << std::setprecision(6) << cpu_view.x[i] << " " << cpu_view.y[i] << " " << cpu_view.z[i]
             << "\n";
    }

    file << "\nPOINT_DATA " << n << "\n";
    file << "SCALARS charge float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n; ++i)
    {
        file << std::scientific << std::setprecision(6) << cpu_view.q[i] << "\n";
    }

    // Embed explicit RGB colors: negative=blue, positive=red, brightness=magnitude
    file << "COLOR_SCALARS rgb 3\n";
    for (size_t i = 0; i < n; ++i)
    {
        float q = cpu_view.q[i];
        float intensity = MIN_INTENSITY + (1.0f - MIN_INTENSITY) * (std::fabs(q) / max_abs_q);
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (q > 0.0f)
            r = intensity;
        else if (q < 0.0f)
            b = intensity;
        else
        {
            r = g = b = MIN_INTENSITY;
        }
        file << std::fixed << std::setprecision(4) << r << " " << g << " " << b << "\n";
    }

    file.close();
}
